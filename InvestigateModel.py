import onnxruntime as ort

# Modeli yükleyelim
try:
    session = ort.InferenceSession("adv_inception_v3_Opset16.onnx")
    
    print("--- MODEL ANALİZ SONUÇLARI ---")
    
    # Giriş Kapısı Bilgileri
    for input_node in session.get_inputs():
        print(f"Giriş Kapısı Adı: {input_node.name}")
        print(f"Beklenen Şekil (Shape): {input_node.shape}")
        print(f"Veri Tipi: {input_node.type}")
    
    print("-" * 30)
    
    # Çıkış Kapısı Bilgileri
    for output_node in session.get_outputs():
        print(f"Çıkış Kapısı Adı: {output_node.name}")
        print(f"Çıkış Şekli: {output_node.shape}")

except Exception as e:
    print(f"Model dosyası bulunamadı veya bir hata oluştu: {e}")