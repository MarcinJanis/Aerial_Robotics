# how to build docker:
> `docker compose -f compose.nogpu.yaml up --build -d`

# how to run docker:
> `docker exec -it cf-container bash`


# to change last character on file windows -> linux: 
sed -i 's/\r$//' setup.sh