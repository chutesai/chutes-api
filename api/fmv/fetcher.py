"""
Fair market value fetcher.
"""

import asyncio
import aiohttp
from functools import lru_cache
from datetime import timedelta
from sqlalchemy import select, func
from loguru import logger
from typing import Dict, Optional
from async_substrate_interface import AsyncSubstrateInterface
from api.config import settings
from api.database import get_session
from api.fmv.schemas import FMV


@lru_cache()
def get_fetcher():
    return FMVFetcher()


class FMVFetcher:
    def __init__(self):
        self.kraken_url = "https://api.kraken.com/0/public/Ticker"
        self.coingecko_url = "https://api.coingecko.com/api/v3/simple/price"
        self.kraken_pairs = {"tao": "TAOUSD"}
        self.coingecko_ids = {"tao": "bittensor"}

    async def store_price(self, ticker: str, price: float):
        """
        Store current FMV in database.
        """
        async with get_session() as session:
            session.add(FMV(ticker=ticker, price=price))
            await session.commit()

    async def get_last_stored_price(
        self, ticker: str, not_older_than: int = None
    ) -> Optional[float]:
        """
        Get the last stored price from database.
        """
        async with get_session() as session:
            query = select(FMV).where(FMV.ticker == ticker)
            if not_older_than is not None:
                query = query.where(FMV.timestamp >= func.now() - timedelta(seconds=not_older_than))
            query = query.order_by(FMV.timestamp.desc()).limit(1)
            result = await session.execute(query)
            fmv = result.scalar_one_or_none()
            if fmv:
                logger.info(
                    f"Fetched stored price from db [{ticker}]: ${fmv.price} @ {fmv.timestamp}"
                )
                return fmv.price
            return None

    async def get_cached_price(self, ticker: str) -> Optional[float]:
        """
        Get current price from redis.
        """
        try:
            cached = await settings.redis_client.get(f"price:{ticker}")
            if cached:
                return float(cached.decode())
        except Exception as e:
            logger.error(f"Error reading from cache: {e}")
        return None

    async def set_cached_price(self, ticker: str, price: float, ttl: int):
        """
        Cache the current price in redis.
        """
        try:
            await settings.redis_client.set(f"price:{ticker}", str(price), ex=ttl)
        except Exception as e:
            logger.error(f"Error writing to cache: {e}")

    async def get_kraken_price(self, ticker: str) -> Optional[float]:
        """
        Get price from Kraken.
        """
        try:
            kraken_ticker = self.kraken_pairs.get(ticker)
            if not kraken_ticker:
                return None
            params = {"pair": kraken_ticker}
            async with aiohttp.ClientSession(raise_for_status=True) as session:
                async with session.get(self.kraken_url, params=params) as response:
                    data = await response.json()
                    if "error" in data and data["error"]:
                        logger.error(f"Kraken API error: {data['error']}")
                        return None
                    result = data["result"]
                    first_pair = next(iter(result.values()))
                    price = float(first_pair["c"][0])
                    return price
        except Exception as e:
            logger.error(f"Error fetching from Kraken: {e}")
            return None

    async def get_coingecko_price(self, ticker: str) -> Optional[float]:
        """
        Get price from CoinGecko.
        """
        try:
            coin_id = self.coingecko_ids.get(ticker)
            if not coin_id:
                return None
            params = {"ids": coin_id, "vs_currencies": "usd"}
            async with aiohttp.ClientSession(raise_for_status=True) as session:
                async with session.get(self.coingecko_url, params=params) as response:
                    data = await response.json()
                    return float(str(data[coin_id]["usd"]))
        except Exception as e:
            logger.error(f"Error fetching from CoinGecko: {e}")
            return None

    async def get_price(self, ticker: str) -> Optional[float]:
        """
        Get crypto price: first trying cache, then kraken, then coingecko, then DB.
        """
        ticker = ticker.lower()
        source = "cache"
        if (cached_price := await self.get_cached_price(ticker)) is not None:
            return cached_price
        if db_price := await self.get_last_stored_price(ticker, not_older_than=3600):
            await self.set_cached_price(ticker, db_price, 60)
            return db_price
        if (price := await self.get_kraken_price(ticker)) is not None:
            source = "kraken"
        if price is None and (price := await self.get_coingecko_price(ticker)):
            source = "coingecko"
        if price is None and (price := await self.get_last_stored_price(ticker)) is not None:
            source = "database"
        if price is not None:
            logger.success(f"Fetched FMV [{ticker}] from {source}: {price}")
            if source != "cache":
                ttl = 60 if source == "database" else 3600
                await self.set_cached_price(ticker, price, ttl)
            if source != "database":
                await self.store_price(ticker, price)
            return price
        logger.error(f"Failed to get FMV for {ticker} from all sources.")
        return None

    async def get_prices(self, tickers: list[str] = ["tao"]) -> Dict[str, Optional[float]]:
        """
        Get prices for multiple tickers concurrently.  A bit of a no-op
        for now since we only actually support tao.
        """
        tasks = [self.get_price(ticker) for ticker in tickers]
        results = await asyncio.gather(*tasks)
        return dict(zip(tickers, results))

    def _alpha_ticker(self, netuid: int) -> str:
        """Get the ticker string for a subnet's alpha token."""
        return f"alpha_{netuid}"

    async def get_subnet_alpha_price(self, netuid: int) -> Optional[float]:
        """
        Get the alpha price (in TAO) for a subnet.
        Returns the amount of TAO required to purchase one unit of Alpha.
        Uses cache -> database fallback like TAO price.
        """
        ticker = self._alpha_ticker(netuid)

        # Try cache first
        try:
            cached = await settings.redis_client.get(f"price:{ticker}")
            if cached:
                return float(cached.decode())
        except Exception as e:
            logger.error(f"Error reading subnet alpha price from cache: {e}")

        # Try database (not older than 1 hour)
        db_price = await self.get_last_stored_price(ticker, not_older_than=3600)
        if db_price is not None:
            await self.set_cached_price(ticker, db_price, 60)
            return db_price

        return None

    async def get_all_subnet_alpha_prices(self) -> Dict[int, float]:
        """
        Get all subnet alpha prices from cache.
        Returns dict mapping netuid -> alpha_price_in_tao.
        """
        prices = {}
        try:
            keys = await settings.redis_client.keys("price:alpha_*")
            if keys:
                values = await settings.redis_client.mget(keys)
                for key, value in zip(keys, values):
                    if value:
                        # Extract netuid from key like "price:alpha_64"
                        netuid = int(key.decode().split("_")[-1])
                        prices[netuid] = float(value.decode())
        except Exception as e:
            logger.error(f"Error reading subnet alpha prices from cache: {e}")
        return prices

    async def fetch_subnet_price_at_block(
        self, substrate, netuid: int, block_hash: str
    ) -> Optional[float]:
        """
        Fetch alpha price for a specific subnet at a specific block.
        Returns alpha_price_in_tao.
        """
        try:
            result = await substrate.runtime_call(
                api="SubnetInfoRuntimeApi",
                method="get_dynamic_info",
                params=[netuid],
                block_hash=block_hash,
            )
            info_data = result if isinstance(result, dict) else (result.value if result else None)
            if not info_data:
                return None

            tao_in = info_data.get("tao_in", 0)
            alpha_in = info_data.get("alpha_in", 0)

            if alpha_in > 0:
                alpha_price = tao_in / alpha_in
            else:
                alpha_price = 1.0

            # Store in database and cache for lazy lookups
            ticker = self._alpha_ticker(netuid)
            await self.store_price(ticker, alpha_price)
            await self.set_cached_price(ticker, alpha_price, 300)

            return alpha_price
        except Exception as e:
            logger.error(f"Error fetching subnet {netuid} price at block {block_hash}: {e}")
            return None

    async def fetch_and_cache_subnet_prices(self) -> Dict[int, float]:
        """
        Fetch all subnet alpha prices from substrate and cache them in redis + database.
        Returns dict mapping netuid -> alpha_price_in_tao.
        Used for lazy background refresh when there are no payment events.
        """
        prices = {}
        try:
            async with AsyncSubstrateInterface(url=settings.subtensor) as substrate:
                result = await substrate.runtime_call(
                    api="SubnetInfoRuntimeApi",
                    method="get_all_dynamic_info",
                    params=[],
                )
                dynamic_infos = (
                    result if isinstance(result, list) else (result.value if result else [])
                )

                for info in dynamic_infos:
                    if info is None:
                        continue
                    info_data = (
                        info
                        if isinstance(info, dict)
                        else (info.value if hasattr(info, "value") else None)
                    )
                    if not info_data:
                        continue

                    netuid = info_data.get("netuid")
                    if netuid is None:
                        continue

                    # tao_in and alpha_in are in rao (10^-9), price = tao_in / alpha_in
                    tao_in = info_data.get("tao_in", 0)
                    alpha_in = info_data.get("alpha_in", 0)

                    if alpha_in > 0:
                        alpha_price = tao_in / alpha_in
                    else:
                        alpha_price = 1.0  # Default to 1:1 if no liquidity

                    prices[netuid] = alpha_price
                    ticker = self._alpha_ticker(netuid)

                    # Store in database
                    await self.store_price(ticker, alpha_price)

                    # Cache with 5 minute TTL
                    await self.set_cached_price(ticker, alpha_price, 300)

                logger.info(f"Cached alpha prices for {len(prices)} subnets")

        except Exception as e:
            logger.error(f"Error fetching subnet prices from substrate: {e}")

        return prices

    async def get_alpha_price_in_usd(self, netuid: int) -> Optional[float]:
        """
        Get the USD price for a subnet's alpha token.
        Calculates: alpha_price_in_tao * tao_price_in_usd
        """
        alpha_price_in_tao = await self.get_subnet_alpha_price(netuid)
        if alpha_price_in_tao is None:
            return None

        tao_price_in_usd = await self.get_price("tao")
        if tao_price_in_usd is None:
            return None

        return alpha_price_in_tao * tao_price_in_usd
