-- One row per check run through the API. Created on first use if missing.

create table if not exists checks (
    id               bigserial primary key,
    created_at       timestamptz not null default now(),
    source           text        not null check (source in ('file', 'photo', 'example', 'json', 'fix')),
    creases          integer     not null check (creases >= 0),
    vertices         integer     not null check (vertices >= 0),
    flat_foldable    boolean     not null,
    failing_vertices integer     not null check (failing_vertices >= 0),
    rules            text[]      not null default '{}',   -- rules broken, one entry per violation
    elapsed_ms       real        not null,
    pattern          jsonb       not null                 -- the cleaned pattern, FOLD format
);

create index if not exists checks_created_at on checks (created_at desc);
