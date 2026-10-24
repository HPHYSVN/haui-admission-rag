# Frontend image: build the Vite app, serve static files with nginx.
# VITE_API_URL is a build-time value. Empty (default) means "same origin",
# which works because nginx reverse-proxies /api to the backend service.

FROM node:22-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

ARG VITE_API_URL=""
ENV VITE_API_URL=$VITE_API_URL

RUN npm run build

FROM nginx:alpine AS runtime

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
