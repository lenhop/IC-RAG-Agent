# Database Assets

This folder stores SQL assets used for UDS database setup and performance tuning.

## Structure

- `uds/create_tables.sql`
  - Creates the UDS schema and tables.
  - Run during initial environment setup.
- `uds/create_indexes.sql`
  - Adds recommended secondary indexes for query acceleration.
  - Run during maintenance windows after validating index impact.

## Usage

- Apply DDL:
  - `clickhouse-client --multiquery < db/uds/create_tables.sql`
- Apply indexes:
  - `clickhouse-client --multiquery < db/uds/create_indexes.sql`

## Notes

- Review SQL statements before running in production.
- Ensure correct database/user permissions.
- Run index changes during low-traffic periods.
