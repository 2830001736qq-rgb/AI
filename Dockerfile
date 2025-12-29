# --- Stage 1: Downloader ---
FROM alpine:latest AS downloader
RUN apk add --no-cache curl tar

WORKDIR /download

# Automatically fetch the latest stable version tag
RUN LATEST_VERSION=$(curl -s https://api.github.com/repos/SagerNet/sing-box/releases/latest | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/' | sed 's/^v//') && \
    curl -L "https://github.com/SagerNet/sing-box/releases/download/v${LATEST_VERSION}/sing-box-${LATEST_VERSION}-linux-amd64.tar.gz" -o sb.tar.gz && \
    tar -xzf sb.tar.gz --strip-components=1

# --- Stage 2: Final Runtime ---
FROM alpine:latest

# Install basic runtime dependencies (ca-certificates for TLS)
RUN apk add --no-cache ca-certificates tzdata bash

WORKDIR /app

# Copy the binary from the downloader stage
COPY --from=downloader /download/sing-box .
# Copy your prepared config
COPY config.json .

# Ensure the binary is executable
RUN chmod +x sing-box

# Northflank default internal port
EXPOSE 8080

# Use the 'exec' form of CMD for better signal handling (graceful shutdown)
CMD ["./sing-box", "run", "-c", "config.json"]
