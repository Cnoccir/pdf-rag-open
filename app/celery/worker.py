from app.web import create_app, db

# Create Flask app instance
flask_app = create_app()

# Initialize Celery
celery_app = flask_app.extensions["celery"]

# Ensure database tables exist
with flask_app.app_context():
    db.create_all()
