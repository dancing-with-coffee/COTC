# Dataset Directory Structure

The new datasets should be organized in the following structure:

```
Software/
├── data/
│   └── dataset/
│       ├── 20newsgroups/
│       │   ├── 20newsgroups.txt
│       │   └── 20newsgroups_labels.txt
│       ├── bbc/
│       │   ├── bbc.txt
│       │   └── bbc_labels.txt
│       ├── reuters8/
│       │   ├── reuters8.txt
│       │   └── reuters8_labels.txt
│       └── webkb/
│           ├── webkb.txt
│           └── webkb_labels.txt
```

## File Format

### Text Files (*dataset*.txt)
- One document per line
- Plain text format
- UTF-8 encoding

### Label Files (*dataset*_labels.txt)
- One label per line (corresponding to the text file)
- Integer labels starting from 1
- The code will automatically convert to 0-based indexing

## Example

If you have 3 documents with 2 classes:

**bbc.txt:**
```
The stock market rose today amid positive earnings reports.
Manchester United won the championship after a thrilling match.
New smartphone features AI-powered camera improvements.
```

**bbc_labels.txt:**
```
1
2
3
```

Where:
- 1 = business
- 2 = sport
- 3 = tech

## Cluster Numbers for Each Dataset
- 20newsgroups: 20 clusters
- BBC: 5 clusters
- Reuters8: 8 clusters
- WebKB: 4 clusters