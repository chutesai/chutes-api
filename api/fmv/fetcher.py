"""
Fair market value fetcher.
"""

import asyncio
import aiohttp
from datetime import timedelta
from sqlalchemy import select, func
from loguru import logger
from typing import Dict, Optional
from api.config import settings
from api.database import SessionLocal
from api.fmv.schemas import FMV


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
        async with SessionLocal() as session:
            session.add(FMV(ticker=ticker, price=price))
            await session.commit()

    async def get_last_stored_price(self, ticker: str, not_older_than: int = None) -> Optional[float]:
        """
        Get the last stored price from database.
        """
        async with SessionLocal() as session:
            query = (
                select(FMV)
                .where(FMV.ticker == ticker)
            )
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
        if (db_price := await self.get_last_stored_price(ticker, not_older_than=3600)):
            await self.set_cached_price(ticker, db_price, 60)
            return db_price
        if (price := await self.get_kraken_price(ticker)) is not None:
            source = "kraken"
        if price is None and (price := await self.get_coingecko_price(ticker)):
            source = "coingecko"
        if (
            price is None
            and (price := await self.get_last_stored_price(ticker)) is not None
        ):
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

    async def get_prices(self, tickers: list[str]) -> Dict[str, Optional[float]]:
        """
        Get prices for multiple tickers concurrently.  A bit of a no-op
        for now since we only actually support tao.
        """
        tasks = [self.get_price(ticker) for ticker in tickers]
        results = await asyncio.gather(*tasks)
        return dict(zip(tickers, results))
