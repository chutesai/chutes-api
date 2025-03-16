import asyncio
import time
from api.config import settings
from loguru import logger
from typing import Dict, Tuple
from dnslib import DNSRecord, QTYPE, RR, A


class RequestHashDNS(asyncio.DatagramProtocol):
    """
    DNS server to check if chutes has processed a particular request.
    """

    def __init__(
        self,
        domain_suffix: str = "reqhash.chutes.ai",
    ):
        self.domain_suffix = domain_suffix
        self.transport = None
        self.request_times: Dict[Tuple[str, int], float] = {}
        self.total_requests = 0
        self.cache_hits = 0

    async def start(self, listen_addr: str = "0.0.0.0"):
        loop = asyncio.get_running_loop()
        self.transport, _ = await loop.create_datagram_endpoint(
            lambda: self, local_addr=(listen_addr, 5303)
        )
        logger.info(f"DNS server started on {listen_addr}:5303")
        logger.info(f"Resolving hostnames ending with '{self.domain_suffix}'")
        asyncio.create_task(self.report_metrics())

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data: bytes, addr: Tuple[str, int]):
        self.request_times[addr] = time.time()
        asyncio.create_task(self.process_dns_query(data, addr))

    async def process_dns_query(self, data: bytes, addr: Tuple[str, int]):
        try:
            request = DNSRecord.parse(data)
            qname = str(request.q.qname)
            if qname.endswith("."):
                qname = qname[:-1]
            if request.q.qtype != QTYPE.A or not qname.endswith(self.domain_suffix):
                logger.debug(f"Ignoring query for {qname} (type={request.q.qtype})")
                return
            request_hash = qname.split(".")[0]
            response_ip = None
            try:
                logger.debug(f"Checking req:{request_hash} key in memcached...")
                value = await settings.memcache.get(f"req:{request_hash}".encode())
                if value is not None:
                    self.cache_hits += 1
                    response_ip = "127.0.0.1"
                    logger.success(f"Cache hit for request {request_hash}")
            except Exception as e:
                logger.error(f"Memcached error: {e}")
            response = request.reply()
            if response_ip:
                response.add_answer(RR(qname, QTYPE.A, ttl=60, rdata=A(response_ip)))
            else:
                logger.info(f"Cache miss for request: {request_hash}")
            response_wire = response.pack()
            self.transport.sendto(response_wire, addr)
            self.total_requests += 1
        except Exception as e:
            logger.error(f"Error processing DNS query: {e}", exc_info=True)

    async def report_metrics(self, interval: int = 60):
        while True:
            await asyncio.sleep(interval)
            if self.total_requests > 0:
                hit_rate = self.cache_hits / self.total_requests
                logger.info(f"Total={self.total_requests}, hit rate={hit_rate}")

    def close(self):
        if self.transport:
            self.transport.close()
        logger.info("DNS server stopped")


async def main():
    server = RequestHashDNS()
    await server.start()
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
