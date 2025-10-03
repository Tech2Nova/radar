
cd radar/

python manage.py runserver

celery -A radar worker -l info
