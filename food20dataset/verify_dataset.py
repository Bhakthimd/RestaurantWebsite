from pathlib import Path
from PIL import Image

# ============================================================
# FOOD20 DATASET VERIFICATION
# ============================================================

# Dataset folders
DATASET_PATH = Path(".")

TRAIN_PATH = DATASET_PATH / "train_set"
TEST_PATH = DATASET_PATH / "test_set"

# Expected number of images
EXPECTED_TRAIN = 70
EXPECTED_TEST = 30

# Supported image formats
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


# ============================================================
# 1. CHECK DATASET FOLDERS
# ============================================================

print("=" * 70)
print("                 FOOD20 DATASET VERIFICATION")
print("=" * 70)

print("\nDataset path:")
print(DATASET_PATH.resolve())

print("\nTrain folder exists :", TRAIN_PATH.exists())
print("Test folder exists  :", TEST_PATH.exists())


# Stop if folders don't exist
if not TRAIN_PATH.exists():
    print("\nERROR: train_set folder not found.")
    exit()

if not TEST_PATH.exists():
    print("\nERROR: test_set folder not found.")
    exit()


# ============================================================
# 2. GET CLASS NAMES
# ============================================================

def get_classes(folder):

    return sorted([
        item.name
        for item in folder.iterdir()
        if item.is_dir()
    ])


train_classes = get_classes(TRAIN_PATH)
test_classes = get_classes(TEST_PATH)


print("\n" + "-" * 70)
print("1. CLASS VERIFICATION")
print("-" * 70)

print("\nNumber of train classes:", len(train_classes))
print("Number of test classes :", len(test_classes))

print("\nTrain classes:")

for class_name in train_classes:
    print("  -", class_name)

print("\nTest classes:")

for class_name in test_classes:
    print("  -", class_name)


# ============================================================
# 3. CHECK CLASS CONSISTENCY
# ============================================================

print("\n" + "-" * 70)
print("2. CLASS CONSISTENCY")
print("-" * 70)

train_class_set = set(train_classes)
test_class_set = set(test_classes)

only_train = train_class_set - test_class_set
only_test = test_class_set - train_class_set

if not only_train and not only_test:

    print("\n✓ Train and test contain the same classes.")

else:

    print("\n✗ Train and test classes do NOT match.")

    if only_train:
        print("\nClasses only in TRAIN:")
        for c in sorted(only_train):
            print("  -", c)

    if only_test:
        print("\nClasses only in TEST:")
        for c in sorted(only_test):
            print("  -", c)


# ============================================================
# 4. GET IMAGE FILES
# ============================================================

def get_images(folder):

    images = []

    for file in folder.rglob("*"):

        if (
            file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ):
            images.append(file)

    return images


# ============================================================
# 5. CHECK TRAIN / TEST IMAGE COUNTS
# ============================================================

print("\n" + "-" * 70)
print("3. IMAGE COUNT VERIFICATION")
print("-" * 70)

all_counts_correct = True

total_train = 0
total_test = 0


print("\nTRAIN / TEST COUNTS:\n")

for class_name in train_classes:

    train_folder = TRAIN_PATH / class_name
    test_folder = TEST_PATH / class_name

    train_images = get_images(train_folder)
    test_images = get_images(test_folder)

    train_count = len(train_images)
    test_count = len(test_images)

    total_train += train_count
    total_test += test_count

    train_status = "✓" if train_count == EXPECTED_TRAIN else "✗"
    test_status = "✓" if test_count == EXPECTED_TEST else "✗"

    print(
        f"{class_name:20} "
        f"Train: {train_count:3} {train_status} | "
        f"Test: {test_count:3} {test_status}"
    )

    if train_count != EXPECTED_TRAIN:
        all_counts_correct = False

    if test_count != EXPECTED_TEST:
        all_counts_correct = False


print("\nTotal training images:", total_train)
print("Total testing images :", total_test)

print("\nExpected training images:", EXPECTED_TRAIN * len(train_classes))
print("Expected testing images :", EXPECTED_TEST * len(test_classes))


if all_counts_correct:

    print(
        "\n✓ Every class contains exactly "
        f"{EXPECTED_TRAIN} training and "
        f"{EXPECTED_TEST} testing images."
    )

else:

    print(
        "\n✗ Some classes do not have the expected "
        f"{EXPECTED_TRAIN}/{EXPECTED_TEST} images."
    )


# ============================================================
# 6. CHECK IMAGE FILE TYPES
# ============================================================

print("\n" + "-" * 70)
print("4. IMAGE FORMAT VERIFICATION")
print("-" * 70)

all_files = []

for folder in [TRAIN_PATH, TEST_PATH]:

    for file in folder.rglob("*"):

        if file.is_file():
            all_files.append(file)


unsupported_files = []

for file in all_files:

    if file.suffix.lower() not in IMAGE_EXTENSIONS:

        unsupported_files.append(file)


print("\nTotal files found:", len(all_files))

print("Supported image files:",
      len(all_files) - len(unsupported_files))

print("Unsupported files:",
      len(unsupported_files))


if unsupported_files:

    print("\nUnsupported files:")

    for file in unsupported_files[:20]:
        print("  -", file)


else:

    print("\n✓ All files use supported image formats.")


# ============================================================
# 7. VERIFY IMAGES ARE NOT CORRUPTED
# ============================================================

def verify_images(folder):

    valid_images = []
    invalid_images = []

    for file in get_images(folder):

        try:

            with Image.open(file) as img:

                # Verify image integrity
                img.verify()

            valid_images.append(file)

        except Exception as e:

            invalid_images.append((file, str(e)))

    return valid_images, invalid_images


print("\n" + "-" * 70)
print("5. IMAGE INTEGRITY VERIFICATION")
print("-" * 70)

print("\nChecking training images...")

train_valid, train_invalid = verify_images(TRAIN_PATH)

print("Checking testing images...")

test_valid, test_invalid = verify_images(TEST_PATH)


print("\nTRAIN")
print("Valid images  :", len(train_valid))
print("Invalid images:", len(train_invalid))

print("\nTEST")
print("Valid images  :", len(test_valid))
print("Invalid images:", len(test_invalid))


# ============================================================
# 8. DISPLAY INVALID IMAGES
# ============================================================

if train_invalid:

    print("\n" + "-" * 70)
    print("INVALID TRAINING IMAGES")
    print("-" * 70)

    for file, error in train_invalid[:20]:

        print("\nFile:", file)
        print("Error:", error)


if test_invalid:

    print("\n" + "-" * 70)
    print("INVALID TESTING IMAGES")
    print("-" * 70)

    for file, error in test_invalid[:20]:

        print("\nFile:", file)
        print("Error:", error)


# ============================================================
# 9. CHECK IMAGE DIMENSIONS
# ============================================================

print("\n" + "-" * 70)
print("6. IMAGE DIMENSION CHECK")
print("-" * 70)


def get_dimensions(folder):

    dimensions = {}

    for file in get_images(folder):

        try:

            with Image.open(file) as img:

                dimensions[file] = img.size

        except:

            pass

    return dimensions


train_dimensions = get_dimensions(TRAIN_PATH)
test_dimensions = get_dimensions(TEST_PATH)


print("\nTraining image dimensions:")

unique_train_dimensions = set(train_dimensions.values())

for dimension in sorted(unique_train_dimensions):

    count = list(train_dimensions.values()).count(dimension)

    print(
        f"  {dimension[0]} x {dimension[1]} : {count} images"
    )


print("\nTesting image dimensions:")

unique_test_dimensions = set(test_dimensions.values())

for dimension in sorted(unique_test_dimensions):

    count = list(test_dimensions.values()).count(dimension)

    print(
        f"  {dimension[0]} x {dimension[1]} : {count} images"
    )


# ============================================================
# 10. CHECK EMPTY CLASS FOLDERS
# ============================================================

print("\n" + "-" * 70)
print("7. EMPTY FOLDER CHECK")
print("-" * 70)

empty_train_classes = []
empty_test_classes = []


for class_name in train_classes:

    if len(get_images(TRAIN_PATH / class_name)) == 0:

        empty_train_classes.append(class_name)


for class_name in test_classes:

    if len(get_images(TEST_PATH / class_name)) == 0:

        empty_test_classes.append(class_name)


if not empty_train_classes:

    print("\n✓ No empty training class folders.")

else:

    print("\n✗ Empty training folders:")

    for c in empty_train_classes:
        print("  -", c)


if not empty_test_classes:

    print("✓ No empty testing class folders.")

else:

    print("✗ Empty testing folders:")

    for c in empty_test_classes:
        print("  -", c)


# ============================================================
# 11. FINAL VERIFICATION
# ============================================================

print("\n")
print("=" * 70)
print("                    FINAL VERIFICATION")
print("=" * 70)

print("\nClasses")
print("-------")
print("Train classes        :", len(train_classes))
print("Test classes         :", len(test_classes))
print("Classes match        :", train_class_set == test_class_set)


print("\nImage Counts")
print("------------")
print("Training images      :", total_train)
print("Testing images       :", total_test)
print("Expected train       :", EXPECTED_TRAIN * len(train_classes))
print("Expected test        :", EXPECTED_TEST * len(test_classes))


print("\nImage Integrity")
print("---------------")
print("Valid train images   :", len(train_valid))
print("Invalid train images :", len(train_invalid))
print("Valid test images    :", len(test_valid))
print("Invalid test images  :", len(test_invalid))


print("\nDataset Status")
print("--------------")


classes_ok = (
    train_class_set == test_class_set
    and len(train_classes) == 10
)

counts_ok = all_counts_correct

integrity_ok = (
    len(train_invalid) == 0
    and len(test_invalid) == 0
)

empty_ok = (
    len(empty_train_classes) == 0
    and len(empty_test_classes) == 0
)


if classes_ok:
    print("✓ Classes: PASS")
else:
    print("✗ Classes: FAIL")


if counts_ok:
    print("✓ Image counts: PASS")
else:
    print("✗ Image counts: FAIL")


if integrity_ok:
    print("✓ Image integrity: PASS")
else:
    print("✗ Image integrity: FAIL")


if empty_ok:
    print("✓ Empty folders: PASS")
else:
    print("✗ Empty folders: FAIL")


print("\n" + "=" * 70)


if classes_ok and counts_ok and integrity_ok and empty_ok:

    print("              ✓ DATASET VERIFICATION PASSED")
    print("=" * 70)

    print("\nYour dataset is ready for preprocessing and training.")

else:

    print("              ✗ DATASET VERIFICATION FAILED")
    print("=" * 70)

    print("\nPlease fix the issues reported above before training.")