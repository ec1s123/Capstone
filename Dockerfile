# Build stage: install the exact npm dependencies from package-lock.json and
# compile the Vite React app into static production files.
FROM node:20-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# Runtime stage: serve the compiled app from Nginx. This keeps the final image
# small and avoids requiring Node.js on the machine that runs the container.
FROM nginx:1.27-alpine AS runtime

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
