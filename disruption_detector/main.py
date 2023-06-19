import os
from app import create_app

if __name__ == '__main__':
    app = create_app()

    host = os.getenv('FLASK_HOST', "0.0.0.0")
    port = os.getenv('FLASK_PORT', 8080)

    app.run(host=host, port=port)
    print(f'Running on {host}:{port}')
