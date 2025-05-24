
# COVID-19 Chest X-Ray Classification using Apache Spark

## 📘 Overview

This project demonstrates a scalable approach to reading and labeling chest X-ray images stored in Hadoop Distributed File System (HDFS), as part of a larger pipeline for COVID-19 and pneumonia detection using deep learning.

We utilize **Apache Spark** to efficiently ingest and preprocess large volumes of image data distributed across different folders representing disease classes.

---

## 💡 Objective

- Load large-scale chest X-ray datasets from HDFS.
- Automatically label the images based on folder structure (COVID, NORMAL, PNEUMONIA).
- Prepare the data for a deep learning classification model.

---

## 📂 Dataset Structure (HDFS)

The dataset is organized in HDFS as:

```
/xray_project/
  ├── COVID/
  ├── NORMAL/
  └── PNEUMONIA/
```

Each folder contains PNG images representing a class.

---

## 🧰 Tools and Technologies

- **Apache Spark**: Distributed data processing engine.
- **HDFS**: Storage system for large-scale datasets.
- **Python (PySpark)**: For writing the processing pipeline.

---

## ⚙️ Steps Executed

1. **Initialize Spark Session**
   ```python
   spark = SparkSession.builder.appName("XRay Dataset From HDFS").getOrCreate()
   ```

2. **Define Label Mapping**
   ```python
   labels = { "COVID": 0, "NORMAL": 1, "PNEUMONIA": 2 }
   ```

3. **Read Images from Each Folder in HDFS**
   ```python
   df = spark.read.format("binaryFile") \
       .option("pathGlobFilter", "*.png") \
       .load(path) \
       .withColumn("label", lit(label))
   ```

4. **Combine All Labeled DataFrames**
   ```python
   full_df = dfs[0]
   for df in dfs[1:]:
       full_df = full_df.union(df)
   ```

5. **Preview the Dataset**
   ```python
   full_df.select("path", "label").show(10, truncate=False)
   ```

---

## 📤 Output

The final Spark DataFrame (`full_df`) contains:
- `path`: HDFS path of the image.
- `label`: Class label (0: COVID, 1: NORMAL, 2: PNEUMONIA)

This output is ready for downstream training using deep learning models (e.g., CNNs in TensorFlow or PyTorch).

---

## 🌍 Contribution to Community

- Helps automate and scale the process of preparing large X-ray datasets.
- Enables integration with deep learning workflows.
- Project can be adapted for other medical imaging tasks.

---

## 📄 License

MIT License – free to use with attribution.
