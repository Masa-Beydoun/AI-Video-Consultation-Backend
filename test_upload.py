import requests

# ✅ غيّر هذا للرابط الصحيح حسب view الرفع عندك
url = 'http://localhost:8000/api/resources/create/'  # تأكد من وجود / في النهاية

# ✅ غيّر هذا لمسار صورة أو ملف موجود فعلاً عندك
file_path = 'C:/Users/Masa/Pictures/test_image.png'

# ✅ أرسل البيانات المرتبطة بالملف
with open(file_path, 'rb') as file:
    files = {'file_path': file}
    data = {'relation_type': 1, 'relation_id': 2}
    response = requests.post(url, files=files, data=data)

# ✅ اطبع النتيجة
print("Status Code:", response.status_code)
print("Response:", response.text)
