# Use Node-RED official image
FROM nodered/node-red:latest

RUN mkdir /data/node-red-kalman-temp /data/node_modules
RUN ln -s /data/node-red-kalman-temp /data/node_modules/

# Set working directory inside the container
WORKDIR /data/node-red-kalman-temp

# Copy package.json and yarn.lock to leverage caching
COPY package.json yarn.lock ./

# Install dependencies using Yarn
RUN yarn install --production --frozen-lockfile

COPY dist/ ./dist/

WORKDIR /usr/src/node-red

# Expose Node-RED default port
EXPOSE 1880

# Start Node-RED
CMD ["yarn", "start", "--", "--userDir", "/data"]

