FROM node:8.5

COPY src/ app/src/
COPY package.json ./app
COPY webpack.config.js ./app
COPY webpack.dev.config.js ./app
COPY webpack.dist.config.js ./app

WORKDIR /app
RUN  npm update --save --dev
RUN  npm install

EXPOSE 3000

CMD  npm run dev
