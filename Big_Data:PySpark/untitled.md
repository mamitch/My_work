# Directions for Running Jupyter Notebooks in Pyspark

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop)
2. Pull the pyspark-notebook image (this takes 10-20 minutes!):
 run in terminal `docker pull jupyter/pyspark-notebook`
3. Start the container with port forwarding (can replace 12345 with anything, but leave 8888 intact):
 run in terminal `docker run -it --rm -p 12345:8888 jupyter/pyspark-notebook`
4. Access the notebook via the URL in the log output, replacing :8888 with the port you chose above (eg :12345)