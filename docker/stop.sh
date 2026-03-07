#!/bin/bash

# UDS Agent Docker Stop Script
# This script stops the Docker containers for the UDS Agent

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
        print_error "Docker is not running."
        exit 1
    fi
}

# Function to check if services are running
check_services_running() {
    if ! docker-compose -f "$DOCKER_COMPOSE_FILE" ps | grep -q "Up"; then
        print_warning "No services are currently running."
        return 1
    fi
    return 0
}

# Function to stop services
stop_services() {
    print_info "Stopping UDS Agent services..."

    docker-compose -f "$DOCKER_COMPOSE_FILE" down

    if [ $? -eq 0 ]; then
        print_success "Services stopped successfully"
    else
        print_error "Failed to stop services"
        exit 1
    fi
}

# Function to stop services and remove volumes
stop_with_volumes() {
    print_warning "This will remove all data volumes!"
    read -p "Are you sure you want to continue? (y/N): " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Stopping services and removing volumes..."

        docker-compose -f "$DOCKER_COMPOSE_FILE" down -v

        if [ $? -eq 0 ]; then
            print_success "Services stopped and volumes removed"
        else
            print_error "Failed to stop services and remove volumes"
            exit 1
        fi
    else
        print_info "Operation cancelled."
        exit 0
    fi
}

# Function to show service status
show_status() {
    print_info "Current Service Status:"
    docker-compose -f "$DOCKER_COMPOSE_FILE" ps
}

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Stop Docker containers for UDS Agent"
    echo ""
    echo "Options:"
    echo "  -v, --volumes    Remove data volumes (WARNING: destroys all data)"
    echo "  -s, --status     Show service status before stopping"
    echo "  -h, --help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                # Stop all services"
    echo "  $0 --volumes      # Stop and remove all data"
    echo "  $0 --status       # Show status then stop"
}

# Main script
main() {
    print_info "UDS Agent Docker Stop Script"
    print_info "============================"

    # Default options
    local remove_volumes=false
    local show_status=false

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--volumes)
                remove_volumes=true
                shift
                ;;
            -s|--status)
                show_status=true
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

    # Show status if requested
    if [ "$show_status" = true ]; then
        show_status
    fi

    # Check if services are running
    if ! check_services_running; then
        exit 0
    fi

    # Stop services
    if [ "$remove_volumes" = true ]; then
        stop_with_volumes
    else
        stop_services
    fi

    print_success "UDS Agent stopped successfully!"
}

# Run main function
main "$@"