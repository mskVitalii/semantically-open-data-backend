import json
import csv
from pathlib import Path
from collections import Counter
import matplotlib

matplotlib.use("Agg")  # Use Agg backend for saving without GUI
import matplotlib.pyplot as plt


def analyze_metadata_fields():
    """
    Analyzes fields from all metadata.json files in subdirectories.
    Counts total fields, unique fields, and duplicate fields.
    """
    # Find all metadata.json files
    current_dir = Path(__file__).parent
    metadata_files = list(current_dir.rglob("*/metadata.json"))

    print(f"Found {len(metadata_files)} metadata.json files\n")

    # Collect all field names
    all_field_names = []

    for metadata_file in metadata_files:
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)

                # Extract field names from "fields" section
                if "fields" in data and isinstance(data["fields"], dict):
                    field_names = list(data["fields"].keys())
                    all_field_names.extend(field_names)
        except Exception as e:
            print(f"Error reading {metadata_file}: {e}")

    # Calculate statistics
    total_fields = len(all_field_names)
    unique_fields = len(set(all_field_names))
    field_counter = Counter(all_field_names)

    # Find duplicate fields (appearing more than once)
    duplicate_fields = {
        field: count for field, count in field_counter.items() if count > 1
    }
    non_duplicate_fields = {
        field: count for field, count in field_counter.items() if count == 1
    }

    # Prepare output lines
    output_lines = [
        "=" * 80,
        "FIELD STATISTICS",
        "=" * 80,
        f"Total fields (from all files): {total_fields}",
    ]
    total_check = len(non_duplicate_fields) + sum(duplicate_fields.values())
    output_lines.append(f"Total check (non-dup + sum of dup counts): {total_check}")
    output_lines.append(f"Unique fields: {unique_fields}")
    output_lines.append(f"Duplicate fields (out of unique): {len(duplicate_fields)}")
    output_lines.append(
        f"Non-duplicate fields (out of unique): {len(non_duplicate_fields)}"
    )
    output_lines.append(f"\n{'=' * 80}")
    output_lines.append("DUPLICATE FIELDS (sorted by frequency)")
    output_lines.append(f"{'=' * 80}\n")

    # Sort duplicate fields by frequency (highest to lowest)
    sorted_duplicates = sorted(
        duplicate_fields.items(), key=lambda x: x[1], reverse=True
    )

    for field_name, count in sorted_duplicates:
        output_lines.append(f"{count:4d}x | {field_name}")

    # Add non-duplicate fields section
    output_lines.append(f"\n{'=' * 80}")
    output_lines.append("NON-DUPLICATE FIELDS (sorted alphabetically)")
    output_lines.append(f"{'=' * 80}\n")

    # Sort non-duplicate fields alphabetically
    sorted_non_duplicates = sorted(non_duplicate_fields.keys())

    for field_name in sorted_non_duplicates:
        output_lines.append(f"   1x | {field_name}")

    # Print to console
    for line in output_lines:
        print(line)

    # Save to file
    output_file = Path(__file__).parent / "datasets_fields_analysis_report.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Found {len(metadata_files)} metadata.json files\n\n")
        f.write("\n".join(output_lines))

    print(f"\n{'=' * 80}")
    print(f"Report saved to: {output_file}")
    print(f"{'=' * 80}")

    return {
        "total_fields": total_fields,
        "unique_fields": unique_fields,
        "duplicate_count": len(duplicate_fields),
        "duplicates": duplicate_fields,
        "sorted_duplicates": sorted_duplicates,
        "field_counter": field_counter,
    }


def plot_field_frequency(stats, top_n=30):
    """
    Creates visualization of field frequency.

    Args:
        stats: analysis results from analyze_metadata_fields()
        top_n: number of top fields to display
    """
    sorted_duplicates = stats["sorted_duplicates"]

    # Get top-N fields
    top_fields = sorted_duplicates[:top_n]
    field_names = [field for field, count in top_fields]
    counts = [count for field, count in top_fields]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

    # Chart 1: Top-N fields (horizontal bar chart)
    ax1.barh(range(len(field_names)), counts, color="steelblue")
    ax1.set_yticks(range(len(field_names)))
    ax1.set_yticklabels(field_names, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("Occurrence count", fontsize=12)
    ax1.set_title(
        f"Top-{top_n} most frequent fields",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(axis="x", alpha=0.3)

    # Add values to bars
    for i, count in enumerate(counts):
        ax1.text(count + 2, i, str(count), va="center", fontsize=8)

    # Chart 2: Frequency distribution histogram
    all_counts = [count for field, count in stats["sorted_duplicates"]]
    ax2.hist(all_counts, bins=50, color="coral", edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Occurrence frequency", fontsize=12)
    ax2.set_ylabel("Number of fields", fontsize=12)
    ax2.set_title("Field frequency distribution", fontsize=14, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_yscale("log")

    # Main title
    fig.suptitle(
        f"Field analysis from {stats['total_fields']} fields in metadata.json files\n"
        f"Unique fields: {stats['unique_fields']}, Duplicates: {stats['duplicate_count']}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    # Save chart
    output_path = Path(__file__).parent / "datasets_fields_frequency_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n{'=' * 80}")
    print(f"Chart saved to: {output_path}")
    print(f"{'=' * 80}")

    plt.close()


def export_fields_to_csv(stats):
    """
    Exports field analysis to CSV file.

    Args:
        stats: analysis results from analyze_metadata_fields()
    """
    output_file = Path(__file__).parent / "datasets_fields_analysis.csv"

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(["Field Name", "Occurrence Count", "Type"])

        # Write duplicate fields (sorted by frequency)
        for field_name, count in stats["sorted_duplicates"]:
            writer.writerow([field_name, count, "Duplicate"])

        # Write non-duplicate fields (sorted alphabetically)
        non_duplicates = sorted(
            [
                field
                for field, count in stats["field_counter"].items()
                if count == 1
            ]
        )
        for field_name in non_duplicates:
            writer.writerow([field_name, 1, "Unique"])

    print(f"\n{'=' * 80}")
    print(f"CSV exported to: {output_file}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    stats = analyze_metadata_fields()
    plot_field_frequency(stats, top_n=50)
    export_fields_to_csv(stats)
