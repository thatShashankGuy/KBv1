Below is a deep dive into **grep**—the classic Unix text-search utility. You’ll find everything from basic invocation to advanced regex, performance tips, and real-world one-liners.

---

## 1. What Is grep?

* **grep** stands for “global regular expression print.”
* It scans files or stdin, printing lines that match a pattern.
* Core implementations: GNU grep (most Linux), BSD grep (macOS), busybox grep (embedded).

Variants:

* `grep` (basic regex),
* `egrep` or `grep -E` (extended regex),
* `fgrep` or `grep -F` (fixed-string),
* `grep -P` (Perl-compatible regex, GNU only).

---

## 2. Invocation & Basic Syntax

```bash
grep [OPTIONS] PATTERN [FILE...]
```

* If no files or file is `-`, reads **stdin**.
* Exit code: `0` if match(es) found; `1` if none; `2` on error.

---

## 3. Pattern Types

1. **Basic Regular Expressions** (BRE) – default.

   * Metacharacters: `\? \+ \| \( \) { }` need backslashes.
2. **Extended Regular Expressions** (ERE) – with `-E` or `egrep`.

   * Metacharacters: `? + | () {}` no escapes.
3. **Fixed-string** – with `-F` or `fgrep`.

   * Treat PATTERN literally; fastest.
4. **Perl-compatible** – with `-P` (GNU grep).

   * Supports lookahead, lookbehind, non-capturing groups.

---

## 4. Key Options

| Option           | Description                                   |
| ---------------- | --------------------------------------------- |
| `-i`             | Ignore case                                   |
| `-v`             | Invert match (print non-matching lines)       |
| `-c`             | Count matching lines per file                 |
| `-n`             | Prefix line number                            |
| `-H` / `-h`      | Show / hide filename on output                |
| `-l` / `-L`      | List files with / without matches             |
| `-r` / `-R`      | Recursive search in directories               |
| `-w`             | Match whole words only                        |
| `-x`             | Match whole lines only                        |
| `-m NUM`         | Stop after NUM matches                        |
| `-C NUM`         | Show NUM context lines before & after match   |
| `-B NUM`         | Show NUM lines before match                   |
| `-A NUM`         | Show NUM lines after match                    |
| `--color[=WHEN]` | Highlight matches (`auto`, `always`, `never`) |
| `-P`             | Perl-compatible regex (GNU only)              |
| `-F`             | Fixed‐string (no regex)                       |
| `-E`             | Extended regex                                |

---

## 5. Basic Examples

```bash
# Simple match
grep 'error' logfile.txt

# Case-insensitive
grep -i 'warning' *.log

# Invert: non-error lines
grep -v 'error' logfile.txt

# Show line numbers
grep -n 'TODO' *.py

# Count occurrences
grep -c 'TODO' *.md
```

---

## 6. Recursive & Filename Control

```bash
# Recursively search C files
grep -R --include='*.c' 'malloc' src/

# Exclude binary
grep -R --binary-files=without-match 'TODO' .

# Hide filenames when grepping single file or stdin
grep -h 'pattern' file.txt
```

---

## 7. Word & Line Anchors

* **Whole-word**: `-w` matches only if PATTERN stands alone.
* **Whole-line**: `-x` matches only if entire line equals PATTERN.
* **Anchors in regex**:

  * `^` start of line,
  * `$` end of line.

```bash
# Lines beginning with “2025”
grep '^2025' log.txt

# Lines ending with “.jpg”
grep '\.jpg$' list.txt

# Word “user” only
grep -w 'user' file
```

---

## 8. Context Lines

```bash
# 3 lines of context around each match
grep -C3 'ERROR' server.log

# Only before
grep -B2 'WARN' app.log

# Only after
grep -A5 'TRACE' app.log
```

---

## 9. Using Extended & Perl Regex

### Extended (`-E` or `egrep`)

```bash
# Match foo or bar
grep -E 'foo|bar' file.txt

# IP address pattern
grep -E '([0-9]{1,3}\.){3}[0-9]{1,3}' access.log
```

### Perl-compatible (`-P`)

```bash
# Lookahead: “foo” not followed by “bar”
grep -P 'foo(?!bar)' file.txt

# Capture groups and backreference
grep -P '([A-Za-z]+) \1' poetry.txt  # repeated word
```

---

## 10. Piping & Combination

```bash
# Find files with .py extension, then search inside
find . -name '*.py' -print0 | xargs -0 grep -n 'import'

# Count distinct error types
grep 'ERROR' app.log | awk '{ print $4 }' | sort | uniq -c

# Filter ps output
ps aux | grep '[n]ginx'
```

> **Tip:** Use the trick `grep '[p]attern'` to avoid matching your own grep process.

---

## 11. Performance Tips

* **Fixed-string (`-F`)** when you don’t need regex—much faster.
* **Limit matches** with `-m NUM`.
* **Narrow search scope**: `--include`/`--exclude`, or `find … -exec grep … +`.
* **Disable color** in scripts: `--color=never`.
* **Use `fgrep`** in very old systems where `-F` is unsupported.

---

## 12. Real-World One-Liners

```bash
# Top 10 most frequent IPs in web log
grep -Eo '([0-9]{1,3}\.){3}[0-9]{1,3}' access.log \
  | sort | uniq -c | sort -nr | head -n10

# List files changed in last 24h containing “TODO”
find . -type f -mtime -1 -print0 | xargs -0 grep -H 'TODO'

# Extract email addresses from text
grep -E -o '\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,6}\b' file.txt

# Count lines in each file with a match, sort by count
grep -R -c 'def ' src/ | sort -t: -k2,2nr
```

---

## 13. Troubleshooting & Tips

* **Unescaped metacharacter** errors → check BRE vs ERE.
* **Binary matches** → use `--binary-files=without-match`.
* **Locale issues** → set `LC_COLLATE=C` for consistent sorting/matching.
* **Huge files** → consider splitting or using `fgrep`.

---

## 14. Cheat-Sheet Summary

| Task                    | Example                           |
| ----------------------- | --------------------------------- |
| Simple grep             | `grep 'foo' file`                 |
| Case-insensitive        | `grep -i 'foo' file`              |
| Whole-word              | `grep -w 'foo' file`              |
| Basic vs extended regex | `grep 'a\+b'` vs `grep -E 'a+b'`  |
| Count matches           | `grep -c 'pattern' file`          |
| Show line numbers       | `grep -n 'pattern' file`          |
| Invert match            | `grep -v 'pattern' file`          |
| Recursive               | `grep -R 'pattern' dir/`          |
| Fixed-string            | `grep -F 'literal*.*' file`       |
| Perl regex              | `grep -P '(?<=foo)bar' file`      |
| Context lines           | `grep -C2 'error' log`            |
| Color highlighting      | `grep --color=always 'warn' file` |

---

