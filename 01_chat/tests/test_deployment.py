"""Deployment artifact checks."""

from pathlib import Path


def test_production_compose_includes_reverse_proxy() -> None:
    """Production Compose should front the API with an Nginx proxy."""

    compose_text = Path("docker-compose.prod.yml").read_text()

    assert "proxy:" in compose_text
    assert 'image: nginx:1.27-alpine' in compose_text
    assert './deploy/nginx/nginx.conf:/etc/nginx/nginx.conf:ro' in compose_text
    assert 'condition: service_healthy' in compose_text
    assert '${AI_CHAT_PORT:-8000}:80' in compose_text


def test_nginx_config_covers_limits_and_streaming() -> None:
    """The Nginx config should enforce basic limits and keep SSE usable."""

    config_text = Path("deploy/nginx/nginx.conf").read_text()

    assert "client_max_body_size 16k;" in config_text
    assert "limit_conn per_ip_connections 20;" in config_text
    assert "limit_req zone=per_ip_requests burst=20 nodelay;" in config_text
    assert "proxy_buffering off;" in config_text
    assert "proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;" in config_text
    assert "proxy_read_timeout 300s;" in config_text
