# Working with Docker containers and Docker Compose

Here is a brief explanation of how the files work that are included here. For more information on **docker** and **docker-compose** feel free to read the friendly manual.

## Creating a Docker Image

In order to run a docker container, you can either pull pre-defined docker images from the internet by specifying their identifier or you can create your own image.

You can define the setup of your image in the `Dockerfile`. The Dockerfile I provided you with is based on NVIDIA's container for tensorflow, which makes setting up GPU support quite easy. Additionally to the base image, it installs SSH, and creates a user with the credentials:

**username: demo**

**password: AAAAAAAA**

You can build the image from the Dockerfile with help of the `docker build` command. Alternatively, you can use the provided `build.sh` script.

The `entrypoint_demo.sh` script is copied into the container and executed on start-up of the container. In this case, it just starts the SSH server and then keeps running.

## Running containers with Docker Compose

Docker compose allows you to specify all the containers you want to run, their network setup and shared disk space. This configuration is stored in the `compose.yaml` file.

As you can see, the file creates only one container, which mounts the home directory within the container to a folder called `home` in your home directory. It also binds its SSH port to the host port `5001`, which means you can connect to the container via SSH as follows:

> ssh -p 5001 demo@127.0.0.1

However, this works only from the **AI06 server** and not from your computer, due to the server's firewall rules.

### Hardware info

Additionally, you can see the hardware resource limitation of the container. They specify that you have access to one GPU (GPU 3), as well as to **16 logical cores** and **128 GB** of RAM. You will need to adjust this if you run more than one container to make sure that you stay within these limits.

You can initially start the docker containers by going into the folder which contains the `compose.yaml` file and then running:

`docker-compose up -d`

You can stop your containers by running (when you are in the same folder as the compose.yaml):

`docker-compose stop`

You can then start the containers again with:

`docker-compose start`
