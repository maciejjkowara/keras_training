# Keras Multi-Output Model: Data Preparation Guide

A visual walkthrough of all the data transformation steps from raw data to model-ready inputs.

---

## 📊 Step 1: Load Raw Data

```python
df = pd.read_excel('mstar_core_bond_index_oct_data.xlsx')
# Shape: (10333, 8)
```

**Raw DataFrame columns:**
```
level_2    level_3              oas  yield  duration  convexity  rating  coupon
Securitized Commercial Mortgage  100  4.82   5.31      0.32      AAA     2.522
Industrials Healthcare           65   4.44   5.01      0.30      BBB1    2.300
...
```

---

## 🎯 Step 2: Separate Features and Targets

### Extract Features
```python
X_numerical = df[['oas', 'yield', 'duration', 'convexity', 'coupon']].values
X_rating = df['rating'].values
```

**Result:**
```
X_numerical: numpy array, shape (10333, 5)
  [[100, 4.82, 5.31, 0.32, 2.522],
   [65,  4.44, 5.01, 0.30, 2.300],
   ...]

X_rating: numpy array, shape (10333,)
  ['AAA', 'BBB1', 'A2', 'AA1', ...]
```

### Extract Targets
```python
y_level2 = df['level_2'].values  # Shape: (10333,)
y_level3 = df['level_3'].values  # Shape: (10333,)
```

**Result:**
```
y_level2: ['Securitized', 'Industrials', 'Financial', ...]
y_level3: ['Commercial Mortgage Backed', 'Healthcare', 'Banking', ...]
```

---

## ✂️ Step 3: Train/Test Split

```python
X_num_train, X_num_test, X_rating_train, X_rating_test, 
y_level2_train, y_level2_test, y_level3_train, y_level3_test = train_test_split(
    X_numerical, X_rating, y_level2, y_level3,
    test_size=0.2,
    stratify=y_level2
)
```

**Result:**
```
Training set: 8266 samples (80%)
Test set:     2067 samples (20%)

X_num_train:    (8266, 5)  - numerical features for training
X_rating_train: (8266,)    - rating strings for training
y_level2_train: (8266,)    - level_2 targets for training
y_level3_train: (8266,)    - level_3 targets for training

(+ corresponding test sets)
```

---

## 🔤 Step 4: Create and Adapt StringLookup Layers

### Create the Lookup Layers
```python
rating_lookup = keras.layers.StringLookup(output_mode="int", num_oov_indices=0)
level2_lookup = keras.layers.StringLookup(output_mode="int", num_oov_indices=0)
level3_lookup = keras.layers.StringLookup(output_mode="int", num_oov_indices=0)
```

### Adapt to Training Data (Learning Phase)
```python
rating_lookup.adapt(X_rating_train)  # Learns vocabulary: {AAA, BBB1, A2, ...}
level2_lookup.adapt(y_level2_train)  # Learns vocabulary: {Securitized, Industrials, ...}
level3_lookup.adapt(y_level3_train)  # Learns vocabulary: {Healthcare, Banking, ...}
```

**What `.adapt()` does:**
- Scans training data to find unique values
- Creates mapping: string → integer (starting from 0)
- Stores vocabulary internally

**Example - Rating Lookup Vocabulary:**
```
'BBB2' → 0
'AAA'  → 1
'A3'   → 2
'BBB1' → 3
'A2'   → 4
...
```

**Vocabulary sizes:**
```
Rating:  10 unique values → indices [0-9]
Level 2: 6 unique values  → indices [0-5]
Level 3: 28 unique values → indices [0-27]
```

---

## 🔢 Step 5: Create and Adapt Normalization Layer

```python
normalizer = keras.layers.Normalization()
normalizer.adapt(X_num_train)
```

**What `.adapt()` does:**
- Calculates mean and std for each of the 5 numerical features
- Stores these statistics (11 non-trainable parameters)

**Example transformation:**
```
Before normalization:
[100, 4.82, 5.31, 0.32, 2.522]  ← raw values with different scales

After normalization:
[0.5, 0.3, -0.2, 0.1, -0.4]  ← standardized: (x - mean) / std
```

---

## 🏗️ Step 6: Build Model Graph (Symbolic Operations)

### Define Input Layers
```python
input_numerical = keras.Input(shape=(5,), name='numerical_features')
input_rating = keras.Input(shape=(1,), dtype='string', name='rating')
```

**These are placeholders (KerasTensors), not actual data:**
```
input_numerical: shape (None, 5)   ← batch_size will be determined later
input_rating:    shape (None, 1)   ← batch_size will be determined later
```

### Apply Preprocessing Layers (Building the Graph)
```python
rating_encoded = rating_lookup(input_rating)
numerical_normalized = normalizer(input_numerical)
```

**These connect layers to inputs:**
```
rating_encoded:        KerasTensor, shape (None, 1), dtype int64
numerical_normalized:  KerasTensor, shape (None, 5), dtype float32
```

**Important:** No actual data flows yet - we're just defining the computation graph!

### Flatten and Combine Features
```python
rating_flat = keras.layers.Flatten()(rating_encoded)
combined_features = keras.layers.Concatenate()([rating_flat, numerical_normalized])
```

**Shape transformations:**
```
rating_encoded:    (None, 1)  ──Flatten──>  rating_flat: (None,)
                                                    |
                                              Concatenate
                                                    |
numerical_normalized: (None, 5) ─────────────>     |
                                                    ↓
                             combined_features: (None, 6)
```

---

## 🎯 Step 7: Prepare Data for Training

### Reshape Rating Inputs (for StringLookup)
```python
X_rating_train.reshape(-1, 1)
```

**Transformation:**
```
Before: (8266,)    ['AAA', 'BBB1', ...]     ← 1D array
After:  (8266, 1)  [['AAA'], ['BBB1'], ...] ← 2D array
```

**Why?** StringLookup expects 2D input: (batch_size, 1)

### Convert to TensorFlow Tensors
```python
X_rating_train_prepared = tf.constant(X_rating_train.reshape(-1, 1), dtype=tf.string)
```

**Why?** TensorFlow backend needs string data in its tensor format, not raw numpy arrays.

### Encode Target Variables
```python
y_level2_train_encoded = np.array(
    level2_lookup(y_level2_train.reshape(-1, 1))
).flatten()
```

**Step-by-step transformation:**
```
1. y_level2_train:                 ['Industrials', 'Securitized', ...]
   Shape: (8266,)

2. .reshape(-1, 1):                [['Industrials'], ['Securitized'], ...]
   Shape: (8266, 1)

3. level2_lookup(...):             [[1], [4], ...]  ← TensorFlow tensor
   Shape: (8266, 1), dtype int64

4. np.array(...):                  [[1], [4], ...]  ← NumPy array
   Shape: (8266, 1)

5. .flatten():                     [1, 4, ...]      ← 1D array
   Shape: (8266,)
```

**Why all these steps?**
- Reshape: StringLookup needs 2D input
- lookup(): Converts strings to integers using learned vocabulary
- np.array(): Converts TensorFlow tensor to NumPy array
- flatten(): Loss function expects 1D labels

**Final prepared data:**
```python
X_num_train_prepared:     (8266, 5)   - numerical features (numpy)
X_rating_train_prepared:  (8266, 1)   - rating strings (TF tensor)
y_level2_train_encoded:   (8266,)     - encoded integers (numpy)
y_level3_train_encoded:   (8266,)     - encoded integers (numpy)
```

---

## 🚀 Step 8: Train the Model

```python
history = model.fit(
    [X_num_train_prepared, X_rating_train_prepared],  # Two inputs (list)
    [y_level2_train_encoded, y_level3_train_encoded], # Two outputs (list)
    validation_split=0.2,
    epochs=20,
    batch_size=32
)
```

**What happens during training:**

1. **Batch creation:** 32 samples at a time from training data
2. **Data flows through model:**
   ```
   Numerical (32, 5) ──> Normalization ──> (32, 5) normalized
                                                |
                                            Concatenate ──> (32, 6)
                                                |
   Rating (32, 1) ──> StringLookup ──> Flatten ──> (32,)
   
   
   Combined (32, 6) ──> Dense(64) ──> Dense(32) ──┬──> level_2: (32, 6)
                                                   └──> level_3: (32, 28)
   ```

3. **Loss calculation:** Compare predictions to true labels for both outputs
4. **Backpropagation:** Update weights to minimize loss
5. **Validation:** Every epoch, evaluate on validation set (last 20% of training data)

---

## 📋 Summary: Data Shape Transformations

### Numerical Features Path
```
Raw data:           (10333, 5)
↓ train_test_split
Training:           (8266, 5)
↓ normalizer.adapt() [learning phase]
↓ normalizer(input)  [in model graph]
Normalized:         (None, 5) [symbolic]
↓ model.fit()
Actual batches:     (32, 5) [runtime]
```

### Rating Feature Path
```
Raw data:           (10333,)     strings
↓ train_test_split
Training:           (8266,)      strings
↓ rating_lookup.adapt() [learning phase]
↓ .reshape(-1, 1)
Reshaped:           (8266, 1)    strings
↓ tf.constant()
TF tensor:          (8266, 1)    TF strings
↓ rating_lookup()   [in model graph]
Encoded:            (None, 1)    integers [symbolic]
↓ Flatten()
Flattened:          (None,)      integers [symbolic]
↓ model.fit()
Actual batches:     (32,)        integers [runtime]
```

### Target Variables Path
```
Raw data:           (10333,)     strings
↓ train_test_split
Training:           (8266,)      strings
↓ level2_lookup.adapt() [learning phase]
↓ .reshape(-1, 1)
Reshaped:           (8266, 1)    strings
↓ level2_lookup()
Encoded tensor:     (8266, 1)    integers
↓ np.array()
Numpy array:        (8266, 1)    integers
↓ .flatten()
Final:              (8266,)      integers [ready for training]
```

---

## 🔑 Key Concepts Explained

### Why Consecutive Indices Starting from 0?

Neural networks need categorical indices to be **consecutive integers starting from 0** for:

1. **One-hot encoding efficiency:**
   ```
   If classes are [0, 1, 2, 3, 4] → 5-dimensional vector
   If classes are [1, 5, 10, 15, 20] → 21-dimensional vector (wasteful!)
   ```

2. **Embedding layer efficiency:**
   ```python
   # Only needs 5 embedding vectors for 5 categories
   Embedding(num_embeddings=5, embedding_dim=8)
   ```

### Learning Phase vs. Transformation Phase

**Learning phase (`.adapt()`):**
- Happens ONCE before building the model
- Uses actual numpy arrays
- Calculates statistics (vocabulary, mean, std)
- Example: `rating_lookup.adapt(X_rating_train)`

**Transformation phase (layer call):**
- Happens in the model graph
- Uses symbolic KerasTensors
- Applies learned transformations
- Example: `rating_lookup(input_rating)`

### Why Reshape, Then Flatten?

```python
# Reshape to 2D for StringLookup
data.reshape(-1, 1)     # (8266,) → (8266, 1)

# StringLookup processes it
encoded = lookup(data)  # (8266, 1)

# Flatten back to 1D for concatenation
encoded.flatten()       # (8266, 1) → (8266,)
```

StringLookup expects 2D input by design, but we need 1D output for combining with other features.

---

## ✅ Final Checklist Before Training

- [x] Data loaded and explored
- [x] Features separated from targets
- [x] Train/test split performed (80/20)
- [x] StringLookup layers created and adapted
- [x] Normalization layer created and adapted
- [x] Model graph built with preprocessing layers
- [x] Model compiled with loss functions and metrics
- [x] Input data reshaped and converted to proper formats
- [x] Target variables encoded to consecutive integers
- [x] Ready to call `model.fit()`!

---

## 🎓 Key Takeaway

**Data preparation is 80% of the work in machine learning!**

The actual model building (defining layers, compiling) is straightforward. The challenge is understanding:
- Data shapes and dimensions
- When to use learning vs. transformation
- Format conversions (numpy ↔ TensorFlow)
- Why preprocessing needs to happen in a specific order

Once your data is properly prepared, training is just one line: `model.fit(X, y)`
