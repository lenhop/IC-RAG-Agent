#!/bin/bash

# UDS Agent Docker Run Script
# This script starts the Docker containers for the UDS Agent

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOCKER_COMPOSE_FILE="docker/docker-compose.uds.yml"

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Function to check if services are already running
check_running() {
    if docker-compose -f "$DOCKER_COMPOSE_FILE" ps | grep -q "Up"; then
        print_warning "Some services are already running. Use './docker/stop.sh' to stop them first."
        read -p "Do you want to continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 0
        fi
    fi
}

# Function to start services
start_services() {
    local profile="${1:-}"

    print_info "Starting UDS Agent services..."

    if [ -n "$profile" ]; then
        print_info "Using profile: $profile"
        docker-compose -f "$DOCKER_COMPOSE_FILE" --profile "$profile" up -d
    else
        docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    fi

    if [ $? -eq 0 ]; then
        print_success "Services started successfully"
    else
        print_error "Failed to start services"
        exit 1
    fi
}

# Function to wait for services to be healthy
wait_healthy() {
    print_info "Waiting for services to be healthy..."

    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if docker-compose -f "$DOCKER_COMPOSE_FILE" ps | grep -q "healthy"; then
            print_success "All services are healthy"
            return 0
        fi

        print_info "Waiting... (attempt $attempt/$max_attempts)"
        sleep 10
        ((attempt++))
    done

    print_warning "Services may still be starting. Check status with: docker-compose -f $DOCKER_COMPOSE_FILE ps"
}

# Function to show service status
show_status() {
    print_info "Service Status:"
    docker-compose -f "$DOCKER_COMPOSE_FILE" ps
}

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Start Docker containers for UDS Agent"
    echo ""
    echo "Options:"
    echo "  -p, --profile PROFILE    Start services with specific profile (with-llm, with-proxy)"
    echo "  -s, --status             Show service status after starting"
    echo "  -w, --wait               Wait for services to be healthy"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                       # Start basic services"
    echo "  $0 --profile with-llm    # Start with LLM service"
    echo "  $0 --profile with-proxy  # Start with reverse proxy"
    echo "  $0 --wait --status       # Start and wait for health, then show status"
}

# Main script
main() {
    print_info "UDS Agent Docker Run Script"
    print_info "==========================="

    # Default options
    local profile=""
    local show_status=false
    local wait_healthy=false

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--profile)
                profile="$2"
                shift 2
                ;;
            -s|--status)
                show_status=true
                shift
                ;;
            -w|--wait)
                wait_healthy=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    # Check prerequisites
    check_docker
    check_running

    # Start services
    start_services "$profile"

    # Wait for health if requested
    if [ "$wait_healthy" = true ]; then
        wait_healthy
    fi

    # Show status if requested
    if [ "$show_status" = true ]; then
        show_status
    fi

    print_success "UDS Agent is running!"
    print_info "API available at: http://localhost:8000"
    print_info "Health check at: http://localhost:8000/api/v1/health"

    if [ -n "$profile" ] && [[ "$profile" == *"proxy"* ]]; then
        print_info "Reverse proxy at: http://localhost"
    fi
}

# Run main function
main "$@"