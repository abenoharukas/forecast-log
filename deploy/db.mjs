/**
 * Database adapter for forecast-log Cloudflare Worker.
 *
 * Backend selection priority:
 *   1. `env.HYPERDRIVE.connectionString` (Supabase Postgres via Cloudflare Hyperdrive)
 *   2. `env.DATABASE_URL` (Direct/Pooler Supabase Postgres URL — useful for local dev or no-Hyperdrive testing)
 *   3. `env.DB` (Cloudflare D1 fallback, kept for easy comparison / rollback)
 *
 * All queries use standard SQL with `?` placeholders; the Postgres adapter rewrites
 * `?` outside single-quoted string literals to `$1, $2, ...`.
 */

import { Pool, types } from "pg";

// Parse Postgres INT8 (BIGINT, COUNT, SUM(int)) and NUMERIC (SUM(bigint)) as JS numbers
// so API payloads match SQLite/Python JSON types exactly.
types.setTypeParser(20, (val) => (val === null ? null : Number(val)));
types.setTypeParser(1700, (val) => (val === null ? null : Number(val)));

/**
 * Rewrite `?` parameter placeholders to `$1, $2, ...`, skipping single-quoted literals.
 */
export function toPgPlaceholders(sql) {
  let out = "";
  let idx = 0;
  let inSingle = false;
  for (let i = 0; i < sql.length; i += 1) {
    const ch = sql[i];
    if (inSingle) {
      out += ch;
      if (ch === "'") {
        if (sql[i + 1] === "'") {
          out += "'";
          i += 1;
        } else {
          inSingle = false;
        }
      }
      continue;
    }
    if (ch === "'") {
      inSingle = true;
      out += ch;
    } else if (ch === "?") {
      idx += 1;
      out += `$${idx}`;
    } else {
      out += ch;
    }
  }
  return out;
}

function isLocalHost(connStr) {
  return /@(localhost|127\.0\.0\.1|\[::1\])[:/]/i.test(connStr);
}

/**
 * Create a per-request database adapter. Lazily opens Postgres pool only if a query runs.
 * Caller MUST call `db.close()` (or `ctx.waitUntil(db.close())`) in `finally`.
 */
export function createDb(env) {
  const hyperdriveUrl = env.HYPERDRIVE && env.HYPERDRIVE.connectionString ? env.HYPERDRIVE.connectionString : "";
  const directUrl = !hyperdriveUrl && env.DATABASE_URL ? String(env.DATABASE_URL).trim() : "";

  if (hyperdriveUrl || directUrl) {
    const connectionString = hyperdriveUrl || directUrl;
    const useHyperdrive = Boolean(hyperdriveUrl);
    let pool = null;

    const getPool = () => {
      if (!pool) {
        const config = {
          connectionString,
          max: 5,
        };
        if (!useHyperdrive && !isLocalHost(connectionString) && !/sslmode=disable/i.test(connectionString)) {
          config.ssl = { rejectUnauthorized: false };
        }
        pool = new Pool(config);
      }
      return pool;
    };

    const query = async (sql, params = []) => {
      const res = await getPool().query(toPgPlaceholders(sql), params);
      return res.rows;
    };

    return {
      dialect: "postgres",
      backend: useHyperdrive ? "hyperdrive" : "postgres",
      async all(sql, params = []) {
        return query(sql, params);
      },
      async first(sql, params = []) {
        const rows = await query(sql, params);
        return rows[0] ?? null;
      },
      async run(sql, params = []) {
        const res = await getPool().query(toPgPlaceholders(sql), params);
        return { changes: res.rowCount ?? 0 };
      },
      async batch(statements) {
        // Run independent read/write statements concurrently through the pool.
        return Promise.all(statements.map(({ sql, params = [] }) => query(sql, params)));
      },
      async execRaw(sql) {
        const res = await getPool().query(sql);
        return { count: res.rowCount ?? null };
      },
      async close() {
        if (pool) {
          const p = pool;
          pool = null;
          await p.end();
        }
      },
    };
  }

  if (env.DB) {
    const d1 = env.DB;
    const prep = (sql, params = []) => {
      const stmt = d1.prepare(sql);
      return params.length ? stmt.bind(...params) : stmt;
    };
    return {
      dialect: "sqlite",
      backend: "d1",
      async all(sql, params = []) {
        const res = await prep(sql, params).all();
        return res.results ?? [];
      },
      async first(sql, params = []) {
        const rows = await this.all(sql, params);
        return rows[0] ?? null;
      },
      async run(sql, params = []) {
        const res = await prep(sql, params).run();
        return { changes: res.meta?.changes ?? 0 };
      },
      async batch(statements) {
        const results = await d1.batch(statements.map(({ sql, params = [] }) => prep(sql, params)));
        return results.map((r) => r.results ?? []);
      },
      async execRaw(sql) {
        const res = await d1.exec(sql);
        return { count: res.count ?? null };
      },
      async close() {},
    };
  }

  throw new Error("No database configured: bind HYPERDRIVE, set DATABASE_URL, or bind D1 DB");
}
