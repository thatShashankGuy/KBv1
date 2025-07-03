# Text Prcoessing Utils

## Contents

* [Basic Building Blocks](#basic-building-blocks)
* [grep – Pattern Search](#grep-–-pattern-search)
* [sed – Stream Editor](#sed-–-stream-editor)
* [awk – Field-oriented Processing](#awk-–-field-oriented-processing)
* [cut – Column Extraction](#cut-–-column-extraction)
* [sort – Ordering Lines](#sort-–-ordering-lines)
* [uniq – Duplicate Removal](#uniq-–-duplicate-removal)
* [tr – Character Translation](#tr-–-character-translation)
* [head & tail – Line/Byte Windows](#head--tail-–-linebyte-windows)
* [wc – Word/Line/Byte Counts](#wc-–-wordlinebyte-counts)
* [paste & join – Merging Columns/Files](#paste--join-–-merging-columnsfiles)
* [xargs – Building Argument Lists](#xargs-–-building-argument-lists)
* [find – File Discovery](#find-–-file-discovery)
* [Combining Utilities](#combining-utilities)

---

## Basic Building Blocks

1. **Standard streams**

   * stdin (file descriptor 0)
   * stdout (fd 1)
   * stderr (fd 2)

2. **Pipelines (`|`)**

   * Chain commands: `cmd1 | cmd2 | cmd3`

3. **Redirection**

   * Output: `> file` (overwrite), `>> file` (append)
   * Input: `< file`

---

## grep – Pattern Search

Search lines in files or stdin matching a regular expression.

```bash
grep [OPTIONS] PATTERN [FILE...]
```

### Key Options

* `-E` Use extended regex (ERE)
* `-F` Fixed-string search (no regex)
* `-i` Ignore case
* `-v` Invert match (lines not matching)
* `-n` Show line numbers
* `-r` Recursive in directories
* `-c` Count matching lines
* `-l` List matching file names only

### Examples

```bash
# Case-insensitive search for "error" in logs
grep -i error /var/log/syslog

# Show lines not containing "DEBUG"
grep -v DEBUG app.log

# Count how many lines contain "TODO"
grep -c TODO *.py

# Recursively search for "main(" in C sources
grep -R --include='*.c' 'main(' src/
```

---

## sed – Stream Editor

Edit text in a pipeline or file, using scripts of commands.

```bash
sed [OPTIONS] 'script' [FILE...]
```

### Common Commands

* `s/RE/REPL/[flags]` Substitute first match per line; `g` for all, `i` for ignore-case
* `d` Delete matching lines
* `p` Print (used with `-n`)
* Addressing: `1,5`, `/PAT/`, `3,$` etc.

### Examples

```bash
# In-place replace "foo"→"bar" in file
sed -i 's/foo/bar/g' file.txt

# Delete blank lines
sed '/^$/d' input.txt

# Print lines 10–20 only
sed -n '10,20p' data.csv
```

---

## awk – Field-oriented Processing

“Swiss army knife” for record/field manipulation.

```bash
awk [OPTIONS] 'pattern { action }' [FILE...]
```

* Default field separator: whitespace; change with `-F`
* `$0` is entire line, `$1`, `$2`, … are fields, `NF` = number of fields
* `BEGIN{}` and `END{}` blocks for pre/post processing

### Examples

```bash
# Print 2nd column of a CSV
awk -F, '{ print $2 }' data.csv

# Sum up 3rd column
awk '{ sum += $3 } END { print sum }' numbers.txt

# Filter lines where $5 > 100
awk '$5 > 100' report.tsv

# Add header then process
awk 'BEGIN{ print "Name\tScore" } { print $1, $3 }' infile
```

---

## cut – Column Extraction

Select only portions of each line.

```bash
cut [OPTIONS] [FILE...]
```

* `-f` Fields (requires `-d` delimiter)
* `-c` Character positions
* `-d` Field delimiter (default: TAB)

### Examples

```bash
# Get 1st and 3rd tab-delimited fields
cut -f1,3 file.tsv

# Characters 5–10 of each line
cut -c5-10 poem.txt

# Use comma delimiter
cut -d, -f2 names.csv
```

---

## sort – Ordering Lines

Sort lines lexically or numerically.

```bash
sort [OPTIONS] [FILE...]
```

### Key Options

* `-n` Numeric sort
* `-r` Reverse
* `-k` Key (field) spec, e.g. `-k3,3`
* `-t` Field delimiter
* `-u` Unique (implies `uniq`)

### Examples

```bash
# Sort numbers in file
sort -n scores.txt

# Sort by 3rd column numerically, tab-delimited
sort -t$'\t' -k3,3n data.tsv

# Reverse alphabetical sort
sort -r names.txt
```

---

## uniq – Duplicate Removal

Filter out repeated lines (requires sorted input for adjacent duplicates).

```bash
uniq [OPTIONS] [INPUT [OUTPUT]]
```

### Options

* `-c` Prefix count of occurrences
* `-d` Only print duplicates
* `-u` Only print unique lines

### Examples

```bash
# Count unique lines
sort items.txt | uniq -c

# Show only duplicates
sort data.log | uniq -d
```

---

## tr – Character Translation

Translate or delete characters.

```bash
tr [OPTIONS] SET1 [SET2]
```

* `-d` Delete SET1
* `-s` Squeeze repeated SET1 into one

### Examples

```bash
# Lowercase → uppercase
echo hello | tr '[:lower:]' '[:upper:]'

# Delete digits
tr -d '0-9' < file.txt

# Collapse multiple spaces to one
tr -s ' ' < input.txt
```

---

## head & tail – Line/Byte Windows

Show first or last N lines/bytes of input.

```bash
head [OPTIONS] [FILE...]
tail [OPTIONS] [FILE...]
```

* Default: 10 lines
* `-n N` lines, `-c N` bytes
* `tail -f` “follow” growth

### Examples

```bash
# First 20 lines
head -n 20 logfile

# Last 50 bytes
tail -c 50 data.bin

# Follow a growing log
tail -f /var/log/syslog
```

---

## wc – Word/Line/Byte Counts

Count lines, words, bytes, characters.

```bash
wc [OPTIONS] [FILE...]
```

* `-l` lines
* `-w` words
* `-c` bytes
* `-m` characters

### Examples

```bash
wc -l *.txt     # count lines in each file
echo foo bar | wc -w   # prints 2
```

---

## paste & join – Merging Columns/Files

* **paste**: merge lines side by side

  ```bash
  paste file1 file2
  paste -d',' f1 f2
  ```
* **join**: relational join on a common field

  ```bash
  sort f1 > s1; sort f2 > s2
  join -t, -1 1 -2 1 s1 s2
  ```

---

## xargs – Building Argument Lists

Turn stdin into command-line arguments.

```bash
command | xargs [OPTIONS] COMMAND [initial-args]
```

* `-n N` max args per COMMAND invocation
* `-P N` parallel runs
* `-0` read null-delimited input (with `-print0`)

### Examples

```bash
# Delete all *.tmp files found
find . -name '*.tmp' -print0 | xargs -0 rm

# Run grep on each .log in parallel (up to 4)
ls *.log | xargs -n1 -P4 grep ERROR
```

---

## find – File Discovery

Recursively locate files/directories by name, type, size, mtime, etc., and run actions.

```bash
find PATH [EXPR …]
```

* `-name PATTERN`
* `-type f|d`
* `-mtime +N` days
* `-exec cmd {} \;` or `-exec … +`

### Examples

```bash
# Find all .sh files
find . -type f -name '*.sh'

# Find and delete empty dirs
find . -type d -empty -delete

# Change perms on *.conf
find /etc -name '*.conf' -exec chmod 644 {} +
```

---

## Combining Utilities

In practice, you’ll chain these tools to build powerful one-liners:

```bash
# List top 10 most common words in a text
tr -cs '[:alpha:]' '\n' < book.txt \
  | tr '[:upper:]' '[:lower:]' \
  | sort \
  | uniq -c \
  | sort -nr \
  | head -n 10

# Extract and sum a field from logs
grep 'ERROR' server.log \
  | awk '{ sum += $NF } END { print "Total:", sum }'
```

---

### Tips & Tricks

* Always quote regex or shell-special characters.
* Use `-r`/`-R` for recursive with grep; `find … -exec … +` for bulk actions.
* When performance matters, prefer specialized tools (e.g. `grep -F` vs. `grep`).
* Test on small data before running destructive operations (e.g. `sed -i`, `rm`).
* Combine `tee` to both view and save intermediate output.
