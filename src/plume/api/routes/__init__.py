from plume.api.routes.forecast import register_forecast_routes
from plume.api.routes.ops import register_ops_routes
from plume.api.routes.service import register_service_routes
from plume.api.routes.sessions import register_session_routes

__all__ = [
    "register_service_routes",
    "register_forecast_routes",
    "register_session_routes",
    "register_ops_routes",
]
