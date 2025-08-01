# connection_debug.py - Add this to your FastAPI application

import logging
import threading
import time
import traceback
import inspect
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from typing import Dict, List
from sqlalchemy import event, Engine
from sqlalchemy.pool import Pool
from fastapi import FastAPI, Request
import asyncio

# Configure logging - QUIET MODE for production
logging.basicConfig(level=logging.WARNING)  # Changed from INFO to WARNING
logger = logging.getLogger("connection_debug")

# Configuration for leak detection
LEAK_DETECTION_CONFIG = {
    'log_connection_creates': False,  # Turn off noisy connection create logs
    'log_session_lifecycle': False,   # Turn off session lifecycle logs
    'log_checkouts': False,          # Turn off checkout/checkin logs
    'connection_age_threshold': 300,  # Log connections older than 5 minutes
    'connection_count_threshold': 20, # Log when active connections exceed this
    'request_leak_threshold': 2,      # Log when request increases connections by this amount
}

def get_filtered_stack_trace(skip_modules=None):
    """
    Get a stack trace filtered to show application code, not library internals.
    """
    if skip_modules is None:
        skip_modules = [
            'connection_debug.py',
            'sqlalchemy/',
            'asyncpg/',
            'site-packages/sqlalchemy',
            'site-packages/asyncpg',
            'uvicorn/',
            'starlette/',
            'anyio/',
            'asyncio/'
        ]
    
    stack = traceback.extract_stack()
    filtered_frames = []
    
    for frame in stack:
        # Skip frames from specified modules
        if not any(skip in frame.filename for skip in skip_modules):
            # Format frame nicely
            frame_info = f"  File \"{frame.filename}\", line {frame.lineno}, in {frame.name}\n    {frame.line}"
            filtered_frames.append(frame_info)
    
    # Return last 8 frames or all if less than 8
    return filtered_frames[-8:] if len(filtered_frames) > 8 else filtered_frames

class ConnectionTracker:
    def __init__(self):
        self.connections = {}
        self.connection_stats = defaultdict(int)
        self.lock = threading.Lock()
        self.request_connections = defaultdict(list)
        
    def track_connection_created(self, connection_id: str, stack_trace: str):
        with self.lock:
            self.connections[connection_id] = {
                'created_at': time.time(),
                'stack_trace': stack_trace,
                'request_id': getattr(threading.current_thread(), 'request_id', None)
            }
            self.connection_stats['created'] += 1
            
            # Only log if we have too many connections or specific issues
            active_count = len(self.connections)
            if active_count > LEAK_DETECTION_CONFIG['connection_count_threshold']:
                logger.warning(f"HIGH CONNECTION COUNT: {active_count} active connections (created: {connection_id})")
            elif LEAK_DETECTION_CONFIG['log_connection_creates']:
                logger.info(f"Connection created: {connection_id}")
            
    def track_connection_closed(self, connection_id: str):
        with self.lock:
            if connection_id in self.connections:
                duration = time.time() - self.connections[connection_id]['created_at']
                
                # Only log long-lived connections or if verbose logging enabled
                if duration > LEAK_DETECTION_CONFIG['connection_age_threshold']:
                    logger.warning(f"LONG-LIVED CONNECTION closed: {connection_id}, duration: {duration:.2f}s")
                elif LEAK_DETECTION_CONFIG['log_connection_creates']:
                    logger.info(f"Connection closed: {connection_id}, duration: {duration:.2f}s")
                
                del self.connections[connection_id]
                self.connection_stats['closed'] += 1
            else:
                # This could indicate a tracking issue
                logger.warning(f"TRACKING ERROR: Attempted to close unknown connection: {connection_id}")
                
    def get_active_connections(self) -> Dict:
        with self.lock:
            current_time = time.time()
            active = {}
            for conn_id, info in self.connections.items():
                active[conn_id] = {
                    **info,
                    'age_seconds': current_time - info['created_at']
                }
            return active
            
    def get_stats(self) -> Dict:
        with self.lock:
            active_count = len(self.connections)
            return {
                **dict(self.connection_stats),
                'active': active_count,
                'leaked': active_count  # Assuming all active are potential leaks
            }

# Global tracker instance
connection_tracker = ConnectionTracker()

# SQLAlchemy event listeners
@event.listens_for(Pool, "connect")
def track_pool_connect(dbapi_conn, connection_record):
    connection_id = f"pool_{id(dbapi_conn)}"
    
    # Get filtered stack trace showing application code
    filtered_frames = get_filtered_stack_trace()
    stack_trace = '\n'.join(filtered_frames)
    
    connection_tracker.track_connection_created(connection_id, stack_trace)

@event.listens_for(Pool, "checkout")
def track_pool_checkout(dbapi_conn, connection_record, connection_proxy):
    connection_id = f"checkout_{id(dbapi_conn)}"
    
    # Only log checkouts if verbose mode enabled
    if LEAK_DETECTION_CONFIG['log_checkouts']:
        logger.info(f"Connection checked out: {connection_id}")

@event.listens_for(Pool, "checkin")
def track_pool_checkin(dbapi_conn, connection_record):
    connection_id = f"checkout_{id(dbapi_conn)}"
    
    # Only log checkins if verbose mode enabled
    if LEAK_DETECTION_CONFIG['log_checkouts']:
        logger.info(f"Connection checked in: {connection_id}")

@event.listens_for(Pool, "close")
def track_pool_close(dbapi_conn, connection_record):
    connection_id = f"pool_{id(dbapi_conn)}"
    connection_tracker.track_connection_closed(connection_id)

# Session tracking decorator
def track_db_session(func):
    """Decorator to track database session usage"""
    async def async_wrapper(*args, **kwargs):
        session_id = f"session_{id(threading.current_thread())}_{time.time()}"
        
        # Only log session lifecycle if verbose mode enabled
        if LEAK_DETECTION_CONFIG['log_session_lifecycle']:
            logger.info(f"Starting session: {session_id} in {func.__name__}")
        
        try:
            result = await func(*args, **kwargs)
            if LEAK_DETECTION_CONFIG['log_session_lifecycle']:
                logger.info(f"Session completed: {session_id}")
            return result
        except Exception as e:
            logger.error(f"Session failed: {session_id}, error: {str(e)}")
            raise
        finally:
            if LEAK_DETECTION_CONFIG['log_session_lifecycle']:
                logger.info(f"Session cleanup: {session_id}")
    
    def sync_wrapper(*args, **kwargs):
        session_id = f"session_{id(threading.current_thread())}_{time.time()}"
        
        # Only log session lifecycle if verbose mode enabled
        if LEAK_DETECTION_CONFIG['log_session_lifecycle']:
            logger.info(f"Starting session: {session_id} in {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            if LEAK_DETECTION_CONFIG['log_session_lifecycle']:
                logger.info(f"Session completed: {session_id}")
            return result
        except Exception as e:
            logger.error(f"Session failed: {session_id}, error: {str(e)}")
            raise
        finally:
            if LEAK_DETECTION_CONFIG['log_session_lifecycle']:
                logger.info(f"Session cleanup: {session_id}")
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

# Middleware to track connections per request
class ConnectionTrackingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request_id = f"req_{id(scope)}_{time.time()}"
            
            # Set request ID in thread local storage (for sync code)
            threading.current_thread().request_id = request_id
            
            # Track initial connection state
            initial_stats = connection_tracker.get_stats()
            
            try:
                await self.app(scope, receive, send)
            finally:
                # Check for connection leaks after request
                final_stats = connection_tracker.get_stats()
                connections_diff = final_stats['active'] - initial_stats['active']
                
                # Only log if there's a significant connection increase
                if connections_diff >= LEAK_DETECTION_CONFIG['request_leak_threshold']:
                    path = scope.get('path', 'unknown')
                    method = scope.get('method', 'unknown')
                    logger.warning(f"CONNECTION LEAK DETECTED: {request_id} - "
                                 f"{method} {path} - +{connections_diff} connections "
                                 f"(total: {final_stats['active']})")
                    
                    # Log details of connections that might be leaked
                    active_conns = connection_tracker.get_active_connections()
                    recent_leaks = []
                    for conn_id, info in active_conns.items():
                        if info.get('request_id') == request_id and info['age_seconds'] < 10:
                            recent_leaks.append(conn_id)
                    
                    if recent_leaks:
                        logger.warning(f"Suspected leaked connections from {request_id}: {recent_leaks}")
                
                # Clean up thread local
                if hasattr(threading.current_thread(), 'request_id'):
                    delattr(threading.current_thread(), 'request_id')
        else:
            await self.app(scope, receive, send)

# Database session context manager with tracking
@contextmanager
def get_tracked_db_session(session_factory):
    """Context manager for database sessions with connection tracking"""
    session_id = f"ctx_session_{time.time()}"
    session = None
    
    try:
        if LEAK_DETECTION_CONFIG['log_session_lifecycle']:
            logger.info(f"Creating tracked session: {session_id}")
        session = session_factory()
        yield session
        
        if LEAK_DETECTION_CONFIG['log_session_lifecycle']:
            logger.info(f"Committing session: {session_id}")
        session.commit()
        
    except Exception as e:
        logger.error(f"Session error in {session_id}: {str(e)}")
        if session:
            if LEAK_DETECTION_CONFIG['log_session_lifecycle']:
                logger.info(f"Rolling back session: {session_id}")
            session.rollback()
        raise
    finally:
        if session:
            if LEAK_DETECTION_CONFIG['log_session_lifecycle']:
                logger.info(f"Closing session: {session_id}")
            session.close()

# Health check endpoint to monitor connections
def create_debug_routes(app, engine):
    @app.get("/debug/connections")
    async def get_connection_debug():
        """Endpoint to check current connection status"""
        try:
            stats = connection_tracker.get_stats()
            active_connections = connection_tracker.get_active_connections()
            
            # Safely get pool info - some attributes might not exist
            pool_info = {}
            try:
                if hasattr(engine, 'pool'):
                    pool = engine.pool
                    pool_info = {
                        'pool_size': getattr(pool, 'size', lambda: 'unknown')(),
                        'checked_in': getattr(pool, 'checkedin', lambda: 'unknown')(),
                        'checked_out': getattr(pool, 'checkedout', lambda: 'unknown')(),
                        'overflow': getattr(pool, 'overflow', lambda: 'unknown')(),
                        'invalid': getattr(pool, 'invalid', lambda: 'unknown')()
                    }
                else:
                    pool_info = {'error': 'No pool attribute on engine'}
            except Exception as pool_error:
                pool_info = {'error': f'Pool info error: {str(pool_error)}'}
            
            # Find long-running connections safely
            long_running = []
            try:
                for conn_id, info in active_connections.items():
                    if info.get('age_seconds', 0) > 60:  # Connections older than 1 minute
                        long_running.append({
                            'id': conn_id,
                            'age_seconds': info['age_seconds'],
                            'request_id': info.get('request_id', 'unknown')
                        })
            except Exception as long_running_error:
                long_running = [{'error': f'Long running check error: {str(long_running_error)}'}]
            
            return {
                'connection_stats': stats,
                'pool_info': pool_info,
                'active_connections_count': len(active_connections),
                'long_running_connections': long_running,
                'engine_info': {
                    'url': str(engine.url).replace(engine.url.password or '', '***') if hasattr(engine, 'url') else 'unknown',
                    'echo': getattr(engine, 'echo', 'unknown')
                }
            }
            
        except Exception as e:
            logger.error(f"Error in connection debug endpoint: {str(e)}")
            return {
                'error': f'Debug endpoint error: {str(e)}',
                'connection_stats': connection_tracker.get_stats(),
                'active_connections_count': len(connection_tracker.get_active_connections())
            }
    
    @app.get("/debug/connections/detailed")
    async def get_detailed_connection_info():
        """Detailed connection information including stack traces"""
        active_connections = connection_tracker.get_active_connections()
        
        detailed_info = []
        for conn_id, info in active_connections.items():
            # Parse stack trace lines
            stack_lines = info['stack_trace'].split('\n') if info['stack_trace'] else []
            
            # Clean up and format stack trace
            formatted_trace = []
            for line in stack_lines:
                if line.strip():
                    formatted_trace.append(line)
            
            detailed_info.append({
                'connection_id': conn_id,
                'age_seconds': info['age_seconds'],
                'request_id': info.get('request_id', 'unknown'),
                'stack_trace': formatted_trace,
                'created_at': info.get('created_at', 0)
            })
        
        return {
            'active_connections': detailed_info,
            'total_count': len(detailed_info)
        }
    
    @app.get("/debug/connections/summary")
    async def get_connection_summary():
        """Summary of connections grouped by creation location"""
        active_connections = connection_tracker.get_active_connections()
        
        # Group by likely creation source
        grouped = defaultdict(list)
        
        for conn_id, info in active_connections.items():
            # Try to identify the source from stack trace
            source = "unknown"
            if info['stack_trace']:
                lines = info['stack_trace'].split('\n')
                for line in lines:
                    if 'def ' in line or 'async def' in line:
                        # Extract function name
                        if ' in ' in line:
                            source = line.split(' in ')[-1].strip()
                            break
                    elif '.py' in line and 'api/' in line:
                        # Extract file and line
                        if 'File "' in line:
                            file_part = line.split('File "')[1].split('"')[0]
                            if 'api/' in file_part:
                                source = file_part.split('api/')[-1]
                                break
            
            grouped[source].append({
                'connection_id': conn_id,
                'age_seconds': info['age_seconds'],
                'request_id': info.get('request_id', 'unknown')
            })
        
        summary = {}
        for source, connections in grouped.items():
            summary[source] = {
                'count': len(connections),
                'avg_age': sum(c['age_seconds'] for c in connections) / len(connections),
                'max_age': max(c['age_seconds'] for c in connections),
                'connections': connections[:3]  # Show first 3 as examples
            }
        
    @app.get("/debug/test")
    async def debug_test():
        """Simple test endpoint to verify debug system is working"""
        try:
            # Test basic tracker functionality
            stats = connection_tracker.get_stats()
            active_count = len(connection_tracker.get_active_connections())
            
            # Test engine access
            engine_type = type(engine).__name__
            
            return {
                'status': 'ok',
                'tracker_stats': stats,
                'active_connections': active_count,
                'engine_type': engine_type,
                'timestamp': time.time()
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }

# Integration functions for easy setup
def setup_connection_debugging(app: FastAPI, engine):
    """
    Easy setup function to add all connection debugging to your FastAPI app.
    Call this after creating your FastAPI app instance.
    """
    # Add the connection tracking middleware
    app.add_middleware(ConnectionTrackingMiddleware)
    
    # Add debug routes
    create_debug_routes(app, engine)
    
    logger.warning("Connection debugging enabled - monitoring for leaks")

def enable_sqlalchemy_logging():
    """
    Enable only essential SQLAlchemy logging for connection debugging.
    """
    # Only enable connection pool warnings and errors
    logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    
    # Ensure our debug logger shows warnings
    logging.getLogger('connection_debug').setLevel(logging.WARNING)
    
    logger.warning("Connection leak detection enabled - only warnings and errors will be logged")

# Wrapper for your existing get_session function
def wrap_get_session(original_get_session):
    """
    Wrapper to add tracking to your existing get_session function.
    
    Usage:
        from api.database import get_session as original_get_session
        get_session = wrap_get_session(original_get_session)
    """
    @contextmanager
    def tracked_get_session(*args, **kwargs):
        session_id = f"wrapped_session_{time.time()}"
        logger.info(f"Creating wrapped session: {session_id}")
        
        try:
            with original_get_session(*args, **kwargs) as session:
                logger.info(f"Session active: {session_id}")
                yield session
                logger.info(f"Session yielded back: {session_id}")
        except Exception as e:
            logger.error(f"Session error in {session_id}: {str(e)}")
            raise
        finally:
            logger.info(f"Session cleanup completed: {session_id}")
    
    return tracked_get_session

# For async session wrapper
def wrap_async_get_session(original_async_get_session):
    """
    Wrapper for async get_session functions.
    """
    @asynccontextmanager
    async def tracked_async_get_session(*args, **kwargs):
        session_id = f"wrapped_async_session_{time.time()}"
        
        if LEAK_DETECTION_CONFIG['log_session_lifecycle']:
            logger.info(f"Creating wrapped async session: {session_id}")
        
        try:
            async with original_async_get_session(*args, **kwargs) as session:
                if LEAK_DETECTION_CONFIG['log_session_lifecycle']:
                    logger.info(f"Async session active: {session_id}")
                yield session
                if LEAK_DETECTION_CONFIG['log_session_lifecycle']:
                    logger.info(f"Async session yielded back: {session_id}")
        except Exception as e:
            logger.error(f"Async session error in {session_id}: {str(e)}")
            raise
        finally:
            if LEAK_DETECTION_CONFIG['log_session_lifecycle']:
                logger.info(f"Async session cleanup completed: {session_id}")
    
    return tracked_async_get_session

# Usage for your main.py:
"""
# Add these imports to your main.py:
from connection_debug import setup_connection_debugging, enable_sqlalchemy_logging, wrap_async_get_session

# Early in your application (before creating engine/app):
enable_sqlalchemy_logging()

# After creating your FastAPI app and engine:
setup_connection_debugging(app, engine)

# Optionally wrap your get_session function:
# from api.database import get_session as original_get_session
# get_session = wrap_async_get_session(original_get_session)
"""