#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


"""
Run python manage.py runserver to start the Django development server.
pip freeze > requirements.     - for requirements.txt
Todo:

Setup CD with github actions to personal server
Can use workflow file for sapsoc website.

docker build -t backend .
'docker run -p 8000:8000 backend'
"""

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
