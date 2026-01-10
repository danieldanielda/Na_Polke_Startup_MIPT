COMPOSE_FILE = ./docker-compose.yml
up:
                docker-compose -f $(COMPOSE_FILE) up --build -d
down:
                docker-compose -f $(COMPOSE_FILE) down
logs:
                docker-compose -f $(COMPOSE_FILE) logs -f

.PHONY: up down logs