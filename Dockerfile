FROM python:3-alpine
EXPOSE 5000

CMD [ "python3", "-m", "http.server", "5000"]