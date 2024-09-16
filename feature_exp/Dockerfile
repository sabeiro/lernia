FROM python:3.6
WORKDIR /app
COPY . /
RUN apt-get update
RUN apt -y install vim libspatialindex-dev cmake redis redis-server git
RUN systemctl status redis-server
RUN pip install -r requirements.txt
ENV LAV_DIR=$APP_DIR
EXPOSE $PORT
EXPOSE 6370-6280
CMD [ "bash", "src/antani/server.sh" ]
