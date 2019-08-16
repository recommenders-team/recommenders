FROM python:3.7.3

#Install FreeTDS and dependencies for PyODBC
RUN apt-get update && apt-get install -y tdsodbc unixodbc-dev unixodbc freetds-dev unixodbc-bin -y  \
 && apt-get clean -y

#Install Microsoft SQL Server
RUN apt-get update && apt-get install -y apt-utils apt-transport-https ca-certificates curl gnupg-agent software-properties-common \
&& curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
&& curl https://packages.microsoft.com/config/debian/9/prod.list > /etc/apt/sources.list.d/mssql-release.list \
&& apt-get update \
&& ACCEPT_EULA=Y apt-get install msodbcsql17 -y

RUN mkdir /code
WORKDIR /code
ADD . /code/
RUN pip install -r requirements.txt

EXPOSE 5000
CMD ["python", "/code/app.py"]