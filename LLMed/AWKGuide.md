Below is a deep dive into **awk**—the field-oriented scripting language that’s indispensable for on-the-fly text processing. Covering everything from basics to advanced features, you’ll find patterns, actions, built-in variables, control flow, functions, performance tips, and real-world examples.

---

## 1. What Is awk?

* **awk** is a small, domain-specific language for processing text files (or streams) composed of records and fields.
* Records default to lines; fields default to whitespace-separated columns.
* You write programs as a sequence of **pattern { action }** pairs; for each input record, if `pattern` matches, `action` runs.

Popular implementations: **gawk** (GNU awk), **mawk**, **nawk**, **busybox awk**.

---

## 2. Invocation & Basic Syntax

```bash
awk [options] 'program' file1 file2 …
```

* **`-F fs`**: set input field separator to `fs` (e.g. `-F,` for CSV).
* **`-v var=val`**: assign AWK variable `var` before execution.
* **`-f script.awk`**: read the program from a file.
* If no files are given or file is `-`, reads **stdin**.

---

## 3. Structure of an awk Program

```awk
# Optional BEGIN block: setup
BEGIN {
    # runs once before any input
}

# Main pattern–action pairs
pattern1 { action1 }
pattern2 { action2 }
…

# Optional END block: teardown
END {
    # runs once after all input
}
```

* **`pattern`**:

  * Regular expression: `/regex/`
  * Numeric/range tests: `$2 > 100`, `NR % 2 == 1`
  * Special patterns: `BEGIN`, `END`, `pattern1,pattern2` (ranges)
  * Omitted: matches every record

* **`action`**: a sequence of statements inside `{ … }`. If omitted, default is `{ print }` (prints `$0`).

---

## 4. Built-in Variables

| Variable      | Meaning                                    |
| ------------- | ------------------------------------------ |
| `NR`          | Number of Records (lines) read so far      |
| `FNR`         | Record number in current file              |
| `NF`          | Number of Fields in current record         |
| `FS`          | Input Field Separator (default whitespace) |
| `OFS`         | Output Field Separator (default space)     |
| `RS`          | Input Record Separator (default `\n`)      |
| `ORS`         | Output Record Separator (default `\n`)     |
| `FILENAME`    | Name of current input file                 |
| `ARGC`/`ARGV` | Command-line argument count/values         |

---

## 5. Field & Record Separators

```bash
# Whitespace fields (default)
awk '{ print $1, $2 }' file

# Comma-separated fields
awk -F, '{ print $3 }' data.csv

# Tab-separated fields
awk -F'\t' '{ print $2 }' table.tsv

# Change record separator: paragraphs as records
awk 'BEGIN{ RS="" } { print "Paragraph:", $0 }' file.txt

# Multi-char FS (GNU awk only)
awk 'BEGIN{ FS="[:,]" } { print $1, $2 }' file
```

---

## 6. Common One-Liners

* **Print specific columns**

  ```bash
  awk '{ print $2, $5 }' file
  ```
* **Sum a column**

  ```bash
  awk '{ sum += $3 } END { print sum }' numbers.txt
  ```
* **Filter rows by field**

  ```bash
  awk '$4 == "PASS"' grades.tsv
  ```
* **Line numbering**

  ```bash
  awk '{ print NR, $0 }' file
  ```
* **Unique count of values**

  ```bash
  awk '{ counts[$1]++ } END { for (v in counts) print v, counts[v] }' file
  ```
* **In-place file edit (GNU awk)**

  ```bash
  gawk -i inplace '{ gsub(/foo/, "bar"); print }' file.txt
  ```

---

## 7. Control Flow & Expressions

```awk
# if/else
$3 > 100 {
    print $1, ">" , $3
} else {
    print $1, "<=", $3
}

# for loop
for (i = 1; i <= NF; i++) {
    sum += $i
}

# while loop
i = 1
while (i <= NF) {
    print $i
    i++
}

# ternary operator
{ print ($2 > 50 ? "High" : "Low") }
```

Comparison and logical operators: `==`, `!=`, `<`, `>`, `<=`, `>=`, `&&`, `||`, `!`.

---

## 8. Arrays & Aggregation

* **Indexed arrays**: `a[1]=…`, `a[2]=…`
* **Associative arrays (keyed by string)**: `count[$1]++`
* Iterate with `for (key in array)`

```awk
# Count occurrences
{ freq[$2]++ }
END {
    for (val in freq)
        print val, freq[val]
}

# Two-dimensional
{ matrix[$1,$2] = $3 }
```

---

## 9. Functions & User-Defined Functions

### Built-ins

* String: `length(s)`, `substr(s,i,n)`, `index(s,t)`, `split(s,a,fs)`, `toupper(s)`, `tolower(s)`
* Math: `sin(x)`, `cos(x)`, `exp(x)`, `log(x)`, `sqrt(x)`, `rand()`, `srand([x])`
* Conversion: `int(x)`, `sprintf(fmt,…)`, `getline`

### Defining your own

```awk
function avg(x, y) {
    return (x + y) / 2
}

{ print "Average:", avg($2, $3) }
```

Place function definitions before use (order doesn’t matter).

---

## 10. Working with Multiple Files

* **FNR** resets per file; **NR** is global.
* Detect file change:

  ```awk
  FNR == 1 { print "== File:", FILENAME, "==" }
  { print }
  ```
* Access ARGV:

  ```awk
  END {
    for (i = 1; i < ARGC; i++)
      print "Arg", i, "=", ARGV[i]
  }
  ```

---

## 11. Advanced gawk Features

* **Multidimensional arrays**
* **Time functions**: `strftime()`, `systime()`
* **Network I/O**: `"/usr/bin/cmd" | getline var`
* **Extensions** via `@include`, dynamic loading (in newer gawk)

---

## 12. Performance Tips

* Use **`-F`** and **`BEGIN{OFS…}`** to avoid repeated splitting/concatenation.
* Prefer **`printf`** over many `print` for precise formatting.
* For large data, use **mawk** when portability allows—it’s faster and lighter.
* Avoid dynamic resizing of huge arrays inside loops; preallocate if possible.

---

## 13. Real-World Examples

### 13.1 Parse Apache Logs

```bash
awk '{
    ip = $1
    # Request in quotes is field 6–8
    req = substr($0, index($0,$6))
    print ip, req
}' access.log
```

### 13.2 CSV to JSON Lite

```bash
awk -F, 'NR==1 {
    for (i=1; i<=NF; i++) header[i]=$i
}
NR>1 {
    printf "{"
    for (i=1; i<=NF; i++) {
        printf "\"%s\":\"%s\"%s", header[i], $i, (i<NF?",":"")
    }
    print "}"
}' data.csv
```

### 13.3 Sliding Window Average

```bash
# 5-line moving average of column 2
{
    vals[NR] = $2
    if (NR >= 5) {
        sum = 0
        for (i = NR-4; i <= NR; i++) sum += vals[i]
        print NR, sum/5
    }
}
```

---

## 14. Debugging & Testing

* **`-n` / `--lint`** (gawk): check script syntax.
* Print debug info:

  ```awk
  { print "DEBUG:", NR, NF, $0 > "/dev/stderr" }
  ```
* Isolate blocks by using `BEGIN{… exit }` or `END{… exit }`.

---

## 15. Cheat-Sheet Summary

| Task                              | Example                                 |
| --------------------------------- | --------------------------------------- |
| Print all lines                   | `awk '{print}' file`                    |
| Filter by regex                   | `awk '/error/' file`                    |
| Print fields 1 & 3                | `awk '{print $1,$3}' file`              |
| Sum field 2                       | `awk '{sum+=$2} END{print sum}' file`   |
| Count lines matching “foo”        | `awk '/foo/{c++} END{print c}' file`    |
| Change FS to `:`, print \$4       | `awk -F: '{print $4}' file`             |
| Replace “old”→“new” and print all | `awk '{gsub(/old/,"new"); print}' file` |

---
