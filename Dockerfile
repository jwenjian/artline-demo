FROM python:3.7

WORKDIR /artline

COPY requirements.txt ./requirements.txt

COPY ArtLine_650.pkl ./ArtLine_650.pkl

COPY app.py ./app.py

COPY templates/* ./templates/

COPY result/* ./result/

COPY tmp/* ./tmp/

COPY core/* ./core/

RUN pip3 --no-cache-dir  --trusted-host=mirrors.aliyun.com install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --verbose

EXPOSE 5000

ENTRYPOINT ["python3","app.py"]
