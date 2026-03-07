#!/bin/bash

# UDS Agent Docker Build Script
# This script builds the Docker containers for the UDS Agent

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="uds-agent"
DOCKER_COMPOSE_FILE="docker/docker-compose.uds.yml"
DOCKERFILE="docker/Dockerfile"

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

# Function to build the UDS Agent image
build_agent() {
    print_info "Building UDS Agent Docker image..."
    docker build -f "$DOCKERFILE" -t "$PROJECT_NAME:latest" .

    if [ $? -eq 0 ]; then
        print_success "UDS Agent image built successfully"
    else
        print_error "Failed to build UDS Agent image"
        exit 1
    fi
}

# Function to build all services
build_all() {
    print_info "Building all Docker services..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" build

    if [ $? -eq 0 ]; then
        print_success "All services built successfully"
    else
        print_error "Failed to build services"
        exit 1
    fi
}

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Build Docker containers for UDS Agent"
    echo ""
    echo "Options:"
    echo "  -a, --all      Build all services (default)"
    echo "  -i, --image    Build only the UDS Agent image"
    echo "  -h, --help     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0              # Build all services"
    echo "  $0 --image      # Build only the agent image"
}

# Main script
main() {
    print_info "UDS Agent Docker Build Script"
    print_info "============================="

    # Check if Docker is running
    check_docker

    # Parse command line arguments
    case "${1:-}" in
        -i|--image)
            build_agent
            ;;
        -a|--all|"")
            build_all
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

    print_success "Build completed successfully!"
    print_info "You can now run the services with: ./docker/run.sh"
}

# Run main function
main "$@"