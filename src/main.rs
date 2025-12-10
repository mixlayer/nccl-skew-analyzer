use anyhow::{Context, Result};
use itertools::Itertools;
use rusqlite::{Connection, OpenFlags};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

#[derive(Debug, Clone)]
struct KernelEvent {
    start: i64,
    end: i64,
    global_pid: i64,
    short_name: String,
    demangled_name: String,
}

fn main() -> Result<()> {
    let db_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "trace0.sqlite".to_string());

    let conn = open_connection(&db_path)?;

    let rank_labels = load_rank_labels(&conn);
    let kernels = load_nccl_kernels(&conn)?;

    if kernels.is_empty() {
        println!("No NCCL kernels found in {db_path}");
        return Ok(());
    }

    let ranks: Vec<i64> = BTreeSet::from_iter(kernels.iter().map(|k| k.global_pid)).into_iter().collect();

    let rank_summary = ranks
        .iter()
        .map(|pid| {
            rank_labels
                .get(pid)
                .cloned()
                .unwrap_or_else(|| format!("globalPid {pid}"))
        })
        .join(", ");

    println!("Ranks (ordered by globalPid): {rank_summary}");
    for (idx, pid) in ranks.iter().enumerate() {
        let label = rank_labels
            .get(pid)
            .cloned()
            .unwrap_or_else(|| format!("globalPid {pid}"));
        println!("  [{idx}] {label}");
    }

    let kernels_by_name = group_by_name_and_rank(&kernels);

    println!("\nNCCL kernels present on all ranks (skew based on kernel start):");
    let mut printed_any = false;
    for (name, per_rank) in kernels_by_name {
        if !ranks.iter().all(|r| per_rank.contains_key(r)) {
            continue;
        }

        let min_len = ranks
            .iter()
            .filter_map(|r| per_rank.get(r))
            .map(|v| v.len())
            .min()
            .unwrap_or(0);
        if min_len == 0 {
            continue;
        }

        printed_any = true;
        println!("\n{name}:");
        for idx in 0..min_len {
            let mut times = Vec::new();
            for r in &ranks {
                if let Some(events) = per_rank.get(r) {
                    if let Some(evt) = events.get(idx) {
                        times.push((*r, evt.start, evt.end, evt.demangled_name.as_str()));
                    }
                }
            }

            let min_t = times.iter().map(|(_, t, _, _)| *t).min().unwrap();
            let max_t = times.iter().map(|(_, t, _, _)| *t).max().unwrap();
            let skew = max_t - min_t;

            println!(
                "  occurrence #{idx}: skew {} ({} ranks)",
                format_duration(skew as i128),
                ranks.len()
            );
            for (r, start, end, demangled) in times {
                let label = rank_labels
                    .get(&r)
                    .cloned()
                    .unwrap_or_else(|| format!("globalPid {r}"));
                println!(
                    "    {label}: +{} from earliest, start={} ns, duration={} ({demangled})",
                    format_duration((start - min_t) as i128),
                    start,
                    format_duration((end - start) as i128),
                );
            }
        }
    }

    if !printed_any {
        println!("  No NCCL kernels appeared on every rank in this trace.");
    }

    Ok(())
}

fn open_connection(path: &str) -> Result<Connection> {
    let p = Path::new(path);
    Connection::open_with_flags(p, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .with_context(|| format!("opening SQLite DB at {}", p.display()))
}

fn load_nccl_kernels(conn: &Connection) -> Result<Vec<KernelEvent>> {
    let mut stmt = conn.prepare(
        r#"
        SELECT
            k.start,
            k.end,
            k.globalPid,
            short.value AS short_name,
            demangled.value AS demangled_name
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds AS short ON k.shortName = short.id
        JOIN StringIds AS demangled ON k.demangledName = demangled.id
        WHERE k.globalPid IS NOT NULL
          AND (short.value LIKE 'nccl%' OR demangled.value LIKE '%nccl%')
        ORDER BY k.start
        "#,
    )?;

    let kernels = stmt
        .query_map([], |row| {
            Ok(KernelEvent {
                start: row.get(0)?,
                end: row.get(1)?,
                global_pid: row.get(2)?,
                short_name: row.get(3)?,
                demangled_name: row.get(4)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

    Ok(kernels)
}

fn load_rank_labels(conn: &Connection) -> BTreeMap<i64, String> {
    let mut labels = BTreeMap::new();
    if let Ok(mut stmt) =
        conn.prepare("SELECT globalPid, pid, name FROM PROCESSES WHERE globalPid IS NOT NULL")
    {
        if let Ok(rows) = stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, Option<i64>>(1)?,
                row.get::<_, Option<String>>(2)?,
            ))
        }) {
            for row in rows.flatten() {
                let label = match (row.1, row.2.as_deref()) {
                    (Some(pid), Some(name)) => format!("pid {pid} ({name})"),
                    (Some(pid), None) => format!("pid {pid}"),
                    (None, Some(name)) => format!("{name}"),
                    (None, None) => format!("globalPid {}", row.0),
                };
                labels.insert(row.0, label);
            }
        }
    }
    labels
}

fn group_by_name_and_rank(
    kernels: &[KernelEvent],
) -> BTreeMap<String, BTreeMap<i64, Vec<KernelEvent>>> {
    let mut grouped: BTreeMap<String, BTreeMap<i64, Vec<KernelEvent>>> = BTreeMap::new();
    for k in kernels {
        grouped
            .entry(k.short_name.clone())
            .or_default()
            .entry(k.global_pid)
            .or_default()
            .push(k.clone());
    }

    for per_rank in grouped.values_mut() {
        for events in per_rank.values_mut() {
            events.sort_by_key(|e| e.start);
        }
    }

    grouped
}

fn format_duration(ns: i128) -> String {
    let abs = ns.abs();
    let (value, unit) = if abs >= 1_000_000_000 {
        (ns as f64 / 1_000_000_000.0, "s")
    } else if abs >= 1_000_000 {
        (ns as f64 / 1_000_000.0, "ms")
    } else if abs >= 1_000 {
        (ns as f64 / 1_000.0, "us")
    } else {
        (ns as f64, "ns")
    };
    format!("{value:.3}{unit}")
}
