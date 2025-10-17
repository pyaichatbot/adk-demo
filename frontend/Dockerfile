# Serve static HTML with a tiny dev server
FROM node:20-alpine
WORKDIR /app
COPY . .
EXPOSE 5173
CMD ["sh", "-c", "npx http-server -p 5173 -c-1 . || (npm i -g http-server && http-server -p 5173 -c-1 .)"]
