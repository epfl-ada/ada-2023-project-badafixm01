with open(R"C:\Users\Berta\Desktop\EPFL\ADA\PROJECT\ada-2023-project-badafixm01\GoEmotions-pytorch\final\chunk_mapping.txt", 'r', encoding='utf-8') as file:
    file_content = file.read()
    print('hi')
# Normalize line breaks, then split
normalized_content = file_content.replace('\r\n', '\n').replace('\r', '\n')
paragraphs = [paragraph for paragraph in normalized_content.split('\n\n') if paragraph]
print('hi')
