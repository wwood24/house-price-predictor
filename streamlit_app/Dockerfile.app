FROM python:3.9

WORKDIR /app

COPY app.py requirements.txt README.md . 

RUN pip install --no-cache -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py","--server.address=0.0.0.0"]