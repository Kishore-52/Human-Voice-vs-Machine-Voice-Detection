import os

print("Real:", len(os.listdir("dataset/real")))
print("Fake:", len(os.listdir("dataset/fake")))