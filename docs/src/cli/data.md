# Data Commands

## store

Store a record in a table.

```bash
axil --db <DB> store <TABLE> '<JSON>'
axil --db ./db store sessions '{"summary": "Fixed auth bug"}'
```

Options: `--agent <name>`, `--entities '<json array>'`, `--llm`

## get

Retrieve a record by ID.

```bash
axil --db <DB> get <ID>
```

## list

List all records in a table.

```bash
axil --db <DB> list <TABLE>
axil --db <DB> list <TABLE> --limit 10
```

## delete

Delete a record by ID.

```bash
axil --db <DB> delete <ID>
```

## update

Update a record's data.

```bash
axil --db <DB> update <ID> '<JSON>'
```

## tables

List all tables with record counts.

```bash
axil --db <DB> tables
```

## legacy alias

`axil insert` remains available as a compatibility alias for `axil store`.

```bash
axil --db ./db insert context '{"type": "architecture", "summary": "..."}'
```
