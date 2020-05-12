FROM node:8.5

COPY node-server/ app/node-server/
COPY package.json ./app

WORKDIR /app
RUN  npm install

CMD  node node-server/server.js
