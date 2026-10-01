class APIError(Exception):

    def __init__(self, status, message, error_type="invalid_request_error", code=None, param=None, headers=None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.error_type = error_type
        self.code = code
        self.param = param
        self.headers = headers or {}

    def to_dict(self):
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


def bad_request(message, param=None, code=None):
    return APIError(400, message, "invalid_request_error", code=code, param=param)


def unauthorized(message="invalid or missing API key"):
    return APIError(401, message, "authentication_error", code="invalid_api_key",
                    headers={"WWW-Authenticate": "Bearer"})


def forbidden(message="this API key is not allowed to perform this action"):
    return APIError(403, message, "permission_error", code="forbidden")


def not_found(message, code="not_found"):
    return APIError(404, message, "invalid_request_error", code=code)


def rate_limited(message, retry_after_s):
    return APIError(429, message, "rate_limit_error", code="rate_limit_exceeded",
                    headers={"Retry-After": str(max(1, int(retry_after_s + 0.999)))})


def overloaded(message="the server is at capacity; retry shortly", retry_after_s=1):
    return APIError(503, message, "server_error", code="server_overloaded",
                    headers={"Retry-After": str(max(1, int(retry_after_s)))})


def internal(request_id):
    return APIError(500, f"internal server error (request id {request_id})", "server_error", code="internal_error")
