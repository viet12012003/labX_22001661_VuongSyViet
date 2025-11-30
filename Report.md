## Báo cáo Lab X: Tổng quan bài toán Text-To-Speech (TTS)

## 1. Giới thiệu chung về bài toán Text-To-Speech

Text-To-Speech (TTS) là bài toán chuyển đổi văn bản thành tín hiệu tiếng nói tự nhiên. Hệ thống TTS hiện đại không chỉ cần "đọc" đúng nội dung mà còn phải:

- **Phát âm chính xác** (đúng từ, đúng âm tiết, đúng trọng âm).
- **Ngữ điệu tự nhiên** (ngữ điệu câu hỏi, câu khẳng định, nhịp nghỉ, lên-xuống giọng).
- **Đa giọng nói, đa ngôn ngữ**, có thể chuyển giọng, giả giọng.
- **Biểu đạt cảm xúc** (vui, buồn, tức giận, trang trọng, ...).
- **Hiệu quả tính toán**, suy luận thời gian thực trên thiết bị hạn chế.

Có thể chia quá trình tiến hóa của TTS thành 3 "level":

- **Level 1 – Rule-based / thống kê cổ điển:** dựa trên luật âm vị, nối đoạn (concatenative) hoặc mô hình tham số (HMM, GMM). Ưu điểm chạy nhanh, ít dữ liệu, dễ port sang nhiều ngôn ngữ nhưng giọng robot, ít tự nhiên.
- **Level 2 – Neural TTS tuỳ biến theo từng người dùng:** dùng mô hình deep learning (Tacotron, FastSpeech, VITS, ...) được fine-tune cho từng người dùng dựa trên tập ghi âm riêng. Tính tự nhiên cao, chi phí suy luận vừa phải.
- **Level 3 – Few-shot / zero-shot TTS:** chỉ cần vài giây ghi âm (hoặc chỉ 1–3 mẫu) để mô phỏng giọng mới. Mô hình lớn, tiêu tốn nhiều tài nguyên nhưng mang lại trải nghiệm cực kỳ linh hoạt.

Các hướng tiếp cận này song song tồn tại, phù hợp với những **bài toán ứng dụng – ràng buộc tài nguyên** khác nhau (thiết bị nhúng, server cloud, ứng dụng trợ lý ảo, giả giọng, giáo dục, game, ...).

---

## 2. Toàn cảnh nghiên cứu TTS

### 2.1. Lược sử phát triển

- **Giai đoạn luật-concatenative (trước deep learning)**  
  Nhiều hệ thống TTS cổ điển dựa trên các luật ngữ âm, hệ thống từ điển phát âm và nối các đơn vị tiếng nói đã được ghi âm sẵn (diphone, triphone, unit selection). Xem trong sách của Taylor [1](#ref1).

- **Giai đoạn thống kê tham số (statistical parametric TTS)**  
  Dùng HMM/GMM để mô hình hoá phổ, F0, độ dài khung, sau đó dùng vocoder (WORLD, STRAIGHT) để tổng hợp tiếng nói. Tiếng nói mượt hơn nhưng có cảm giác "metallic" và kém tự nhiên.

- **Thời kỳ neural TTS (Level 2)**  
  DeepMind công bố WaveNet [2](#ref2), Google/DeepMind ra Tacotron, Tacotron 2 [3](#ref3); Microsoft, Baidu, ... đề xuất Deep Voice, FastSpeech, FastPitch.  
  Gần đây xuất hiện mô hình end-to-end như VITS [4](#ref4) kết hợp text → mel + vocoder trong một mạng duy nhất.

- **Thời kỳ few-shot / zero-shot (Level 3)**  
  Các công trình như YourTTS [5](#ref5), VALL-E [6](#ref6), Voicebox [7](#ref7) dùng kiến trúc encoder–decoder lớn, mã hoá giọng thành embedding và sinh tiếng nói mới từ vài giây ví dụ.

- **Xu hướng mới: Expressive & controllable TTS, watermarking & chống deepfake**  
  Nhiều survey (ví dụ Anusuya & Katti, 2024 [8](#ref8); IJCRT 2025 [9](#ref9)) nhấn mạnh việc thêm điều khiển cảm xúc, phong cách, độ nói nhanh–chậm, cũng như nghiên cứu watermark hoá đầu ra để tránh lạm dụng deepfake.

### 2.2. Kiến trúc pipeline tổng quát

Một hệ thống TTS điển hình (đặc biệt Level 2–3) gồm các khối:

- **Tiền xử lý văn bản (Text Front-end)**  
  Chuẩn hóa văn bản (chữ số → chữ, mở rộng viết tắt), tách câu, gán nhãn ngữ pháp, chuyển sang chuỗi phoneme hoặc grapheme.

- **Mô hình acoustic / seq2seq**  
  Dự đoán đặc trưng trung gian như mel-spectrogram hoặc mã âm thanh (codec token). Ví dụ: Tacotron/Transformer TTS (mel), VITS (kết hợp variational + flow), VALL-E (codec LM).

- **Vocoder**  
  Chuyển mel-spectrogram/codec token thành sóng âm. Từ Griffin–Lim cổ điển đến WaveNet, WaveGlow, HiFi-GAN, ...

- **Bộ điều khiển giọng nói & cảm xúc**  
  Thêm embedding cho loa (speaker embedding), style token hoặc vector cảm xúc để điều khiển pitch, năng lượng, tốc độ.

---

## 3. Level 1 – TTS dựa trên luật và mô hình cổ điển

### 3.1. Nguyên lý

Các hệ thống Level 1 thường:

- Dùng **mô tả luật phát âm** (rule-based): chuyển từ → chuỗi phoneme theo luật ngữ âm.
- Với TTS nối đoạn (concatenative): chọn các đoạn speech sẵn (diphone/unit selection) rồi ghép lại theo ngữ điệu đã thiết kế.
- Với TTS tham số thống kê: sử dụng HMM/GMM để sinh chuỗi tham số tiếng nói, sau đó vocoder tổng hợp lại.

### 3.2. Ưu điểm

- **Tốc độ suy luận rất nhanh**, chạy tốt trên thiết bị yếu (embedded, thiết bị IoT cũ).
- **Không yêu cầu GPU**; dễ deploy trên hệ thống cũ.
- **Dễ mở rộng đa ngôn ngữ** nếu có bộ luật & từ điển phát âm (dù xây dựng những thứ này tốn công ban đầu, nhưng về sau ít tốn dữ liệu lớn).
- Có thể đảm bảo **độ kiểm soát cao** (ví dụ: luôn đọc số điện thoại, địa chỉ theo format rất chuẩn, ổn định).

### 3.3. Nhược điểm

- **Độ tự nhiên thấp**, thường nghe "robot", ít cảm xúc, ngữ điệu đơn điệu.
- Khó mô hình các hiện tượng tự nhiên như nuốt âm, nối âm, nhấn nhá tinh tế.
- **Chi phí xây dựng luật cao**, cần nhiều chuyên gia ngữ âm và ngôn ngữ học cho từng ngôn ngữ.

### 3.4. Trường hợp sử dụng phù hợp

- **Ứng dụng nhúng**: thiết bị nhỏ, ít tài nguyên (máy đo, thiết bị đọc thông báo, thiết bị hỗ trợ người khiếm thị đời cũ).
- **Hệ thống cần độ ổn định cao, ít thay đổi**: IVR truyền thống trong tổng đài, đọc số tài khoản, OTP, thông báo giao dịch.
- **Ngôn ngữ ít tài nguyên**: chưa có dữ liệu đủ lớn cho neural TTS.

### 3.5. Cách tối ưu pipeline Level 1 trong nghiên cứu hiện nay

- Kết hợp **rule-based + statistical TTS** để cải thiện phần prosody.
- Dùng **pretrained vocoder hiện đại** (HiFi-GAN, WaveRNN) cho phần tổng hợp cuối để tăng tự nhiên.
- Dùng các phương pháp **post-processing** (equalization, noise shaping) để giảm cảm giác "metallic" của vocoder cổ điển.

---

## 4. Level 2 – Neural TTS fine-tune cho từng người dùng

### 4.1. Nguyên lý

Neural TTS thường gồm 2 bước:

- **Text → mel-spectrogram (acoustic model)**: dùng kiến trúc seq2seq, attention (Tacotron 2), transformer (FastSpeech, FastSpeech 2), VAE/flow (VITS).
- **Mel → sóng âm (vocoder)**: WaveNet, WaveGlow, Parallel WaveGAN, HiFi-GAN, ...

Trong setup **tuỳ biến cho từng người dùng**, nhà cung cấp có:

- Một **mô hình nền (base model)** được huấn luyện trên dữ liệu đa loa, đa ngôn ngữ.
- Người dùng ghi âm một lượng dữ liệu vừa phải (vài chục phút đến vài giờ) → fine-tune toàn bộ hoặc một phần (speaker adapter, LoRA, prefix-tuning) để tạo giọng riêng.

### 4.2. Ưu điểm

- **Độ tự nhiên rất cao**, gần với giọng người thật, đặc biệt khi dữ liệu fine-tune đủ sạch.
- Với mỗi người dùng, có thể tạo **giọng riêng nhất quán** (tên thương hiệu, KOL, MC ảo, ...).
- Chi phí suy luận **thấp hơn Level 3**, có thể deploy trên máy chủ bình thường hoặc thậm chí edge device với model nhẹ.

### 4.3. Nhược điểm

- **Cần dữ liệu ghi âm riêng cho từng người dùng**, tương đối tốn công: chuẩn bị script, phòng thu, làm sạch dữ liệu.
- Nếu mỗi người một model, **quản lý nhiều checkpoint** là gánh nặng: lưu trữ và versioning.
- Vẫn chưa thực sự few-shot: thường cần tối thiểu hàng chục phút đến vài giờ ghi âm để đạt chất lượng cao.

### 4.4. Trường hợp sử dụng phù hợp

- **Thương hiệu, doanh nghiêp** muốn có giọng riêng (giọng tổng đài, giọng trợ lý ảo, MC ảo) với chất lượng cao.
- **Ứng dụng giáo dục, audiobook, e-learning**: cần chất lượng tự nhiên, ổn định, nhưng số lượng giọng không quá nhiều.
- **Ứng dụng on-premise**: muốn giữ dữ liệu giọng nói người dùng trong hạ tầng riêng.

### 4.5. Cách xây pipeline tối ưu cho Level 2

- Dùng **base model đa loa, đa ngôn ngữ** được huấn luyện trước (pretrained) → giảm dữ liệu cần cho mỗi người dùng.
- Áp dụng **speaker embedding** hoặc **speaker adapter** để không cần fine-tune toàn bộ mạng, chỉ học một số lớp nhỏ → tiết kiệm tài nguyên và dễ quản lý.
- Kết hợp **data augmentation** (thay đổi tốc độ, pitch, thêm noise nhẹ) giúp giảm overfitting khi dữ liệu người dùng ít.
- Tách hệ thống thành **service microservice**: front-end text → phoneme, acoustic model, vocoder, caching kết quả cho đoạn lặp lại.

---

## 5. Level 3 – Few-shot / zero-shot TTS

### 5.1. Nguyên lý

Mục tiêu Level 3 là: **chỉ từ vài giây tiếng nói mẫu**, hệ thống đã có thể bắt chước giọng mới (timbre, accent, prosody) mà **không cần fine-tune riêng lẻ**.

Các mô hình điển hình:

- **YourTTS** [5](#ref5): 
  - Huấn luyện trên tập đa ngôn ngữ lớn.
  - Học embedding loa từ mẫu speech đầu vào.
  - Cho phép zero-shot cross-lingual TTS: dùng giọng của người nói tiếng này để nói ngôn ngữ khác.

- **VALL-E** [6](#ref6):
  - Sử dụng discrete codec (như EnCodec) để mã hoá speech thành token.
  - Huấn luyện mô hình language model trên chuỗi token (tương tự GPT cho audio).
  - Given vài giây prompt, mô hình sinh tiếp đoạn tiếp theo giữ nguyên giọng.

- **Voicebox** [7](#ref7) và các hệ thống tương tự:  
  Tập trung vào tổng hợp speech linh hoạt (speech infilling, style transfer) dựa trên generative model lớn.

### 5.2. Ưu điểm

- **Trải nghiệm người dùng tối ưu**: chỉ cần tải lên vài giây âm thanh là có giọng mới.
- Hỗ trợ **rất nhiều giọng, nhiều accent** mà không phải fine-tune từng người.
- Tiềm năng lớn trong **đa ngôn ngữ**: zero-shot cross-lingual TTS.

### 5.3. Nhược điểm

- **Mô hình cực lớn, đòi hỏi tài nguyên tính toán cao** (GPU mạnh, bộ nhớ lớn) cho cả huấn luyện lẫn suy luận.
- Hệ thống phức tạp, bao gồm codec, LM cho audio, alignment giữa text và audio.
- Nguy cơ **lạm dụng deepfake** cao, khó phân biệt giọng thật – giả nếu không có watermark.

### 5.4. Trường hợp sử dụng phù hợp

- **Dịch vụ cloud TTS cao cấp**: cung cấp cho rất nhiều người dùng với nhiều kiểu giọng mà không lưu trữ model riêng cho từng người.
- **Ứng dụng sáng tạo nội dung**: video, phim, game, demo ý tưởng nhanh.
- **Nghiên cứu về khả năng tổng quát hoá của mô hình ngôn ngữ đa phương thức**.

### 5.5. Hướng tối ưu pipeline Level 3

- Dùng **codec hiệu quả** (EnCodec, SoundStream) để giảm chiều dữ liệu audio và tăng tốc suy luận.
- Tách mô hình thành phần **speaker encoder** (trích xuất embedding từ vài giây speech) + **content model** (LM hoặc seq2seq) để dễ transfer.
- Áp dụng **distillation** hoặc **model compression** (quantization, pruning) để đưa model lớn về phiên bản nhẹ cho suy luận.
- Tích hợp **cơ chế bảo vệ**: watermark, kiểm tra tính hợp lệ của mẫu đầu vào (bỏ các tấn công audio adversarial).

---

## 6. So sánh 3 level và các chiến lược tối ưu

### 6.1. So sánh ưu/nhược điểm

- **Level 1 (Rule-based / thống kê)**
  - Ưu: nhanh, nhẹ, ít phụ thuộc dữ liệu, phù hợp đa nền tảng cũ, dễ kiểm soát.
  - Nhược: giọng robot, ít tự nhiên, khó thêm cảm xúc phức tạp.

- **Level 2 (Neural TTS fine-tune per user)**
  - Ưu: rất tự nhiên, giữ được đặc trưng giọng từng người, chi phí suy luận tương đối thấp.
  - Nhược: cần dữ liệu ghi âm riêng, quản lý nhiều model, không truly few-shot.

- **Level 3 (Few-shot / zero-shot)**
  - Ưu: chỉ cần vài giây mẫu, cực linh hoạt, nhiều giọng, hỗ trợ đa ngôn ngữ.
  - Nhược: mô hình lớn, đắt đỏ về tài nguyên, rủi ro deepfake cao.

### 6.2. Lựa chọn level theo bài toán thực tế

- **Thiết bị nhúng, IoT, hệ thống đọc thông báo đơn giản** → ưu tiên **Level 1**.
- **Doanh nghiêp muốn giọng thương hiệu riêng, chất lượng cao, không đổi quá thường xuyên** → **Level 2**.
- **Nền tảng cloud, sản phẩm sáng tạo nội dung, demo concept nhanh, cần nhiều giọng linh hoạt** → **Level 3**.

### 6.3. Các chiến lược giảm nhược điểm, tối đa ưu điểm

- **Kết hợp hybrid**: dùng Level 1 cho phần nội dung rất chuẩn hoá (số, ký hiệu đặc biệt) và Level 2/3 cho câu tự nhiên.
- **Transfer learning & adapter**: với Level 2, tránh fine-tune toàn model, chỉ học thêm adapter/speaker embedding để tiết kiệm tài nguyên.
- **Model compression**: distillation, quantization để đưa các mô hình Level 2/3 xuống thiết bị edge.
- **Cache & streaming**: đối với ứng dụng realtime, cache các cụm thường lặp lại, sử dụng streaming TTS để giảm độ trễ cảm nhận.

---

## 7. Khía cạnh đạo đức và watermark trong TTS

### 7.1. Rủi ro đạo đức

Neural TTS, đặc biệt Level 3, đem lại nguy cơ:

- **Deepfake giọng nói**: giả giọng người nổi tiếng, lãnh đạo, người thân để lừa đảo.
- **Tin giả (misinformation)**: tạo các đoạn speech "phát biểu" mà nạn nhân chưa từng nói.
- **Xâm phạm quyền riêng tư & quyền đối với giọng nói**: sử dụng dữ liệu ghi âm mà không xin phép.

Các tổ chức nghiên cứu và doanh nghiêp đề xuất nhiều biện pháp giảm thiểu:

- Yêu cầu **chứng thực danh tính** trước khi tạo giọng.
- **Giới hạn API**, kiểm tra mục đích sử dụng, tự động phát hiện nội dung nhạy cảm.
- Hợp tác với nền tảng mạng xã hội và cơ quan báo chí để **gắn nhãn nội dung tổng hợp**.

### 7.2. Watermark hoá tiếng nói tổng hợp

Watermark là kỹ thuật **nhúng tín hiệu bí mật** vào đầu ra TTS để máy có thể phát hiện nội dung là do AI sinh ra, trong khi tai người gần như không nhận ra.

- **Watermark tín hiệu audio**: chèn mẫu tín hiệu ở miền tần số/thời gian sao cho bền vững với nén, truyền qua mạng nhưng không làm giảm chất lượng cảm nhận.  
  Tham khảo các nghiên cứu watermark audio tổng quát trong sách Cox et al., 2008 [10](#ref10).

- **Watermark ở mức mô hình / token**: với các hệ thống dựa trên token (codec token, mel token), có thể áp dụng ý tưởng tương tự **watermark cho LLM** (như trong công trình của Kirchenbauer et al., 2023 [11](#ref11)) để điều chỉnh phân phối xác suất sinh token theo một pattern bí mật.

- **Kết hợp watermark + detector**: thiết kế riêng bộ phát hiện sử dụng watermark để tăng độ chính xác, đồng thới huấn luyện mô hình chống lại các thao tác xoá watermark (nén, thêm nhiễu, chỉnh pitch).

### 7.3. Khuyến nghị thực hành tốt

- Luôn **thông báo rõ** cho người dùng cuối khi đang nghe giọng tổng hợp.
- Cho phép **cá nhân/nhãn hàng thu hồi quyền sử dụng giọng**, yêu cầu xoá dữ liệu huấn luyện.
- Tích hợp **watermark bắt buộc** trong các hệ thống TTS thương mại quy mô lớn.

---

## 8. Kết luận

Bài toán Text-To-Speech đã đi một chặng đường dài từ các hệ thống rule-based, concatenative (Level 1) đến neural TTS tuỳ biến theo từng người (Level 2) và few-shot/zero-shot TTS (Level 3). Mỗi hướng tiếp cận có ưu/nhược điểm và phù hợp với các trường hợp sử dụng khác nhau, tuỳ thuộc vào:

- Yêu cầu **tự nhiên và cảm xúc** của tiếng nói.
- **Tài nguyên tính toán** và khả năng triển khai.
- **Mức độ cá nhân hoá** và số lượng giọng cần hỗ trợ.

Các hướng nghiên cứu hiện nay tập trung vào:

- Tăng độ tự nhiên, giàu cảm xúc và khả năng điều khiển.
- Giảm chi phí suy luận, hỗ trợ realtime, đa thiết bị.
- Hỗ trợ đa ngôn ngữ, zero-shot, cross-lingual.
- Đảm bảo khía cạnh **đạo đức, an toàn**, bao gồm watermark và chống deepfake.

Trong tương lai gần, TTS có khả năng trở thành một thành phần mặc định trong nhiều hệ thống giao tiếp người–máy, với giọng nói tuỳ chỉnh, an toàn và dễ tiếp cận cho mọi người dùng.

---

## Tài liệu tham khảo

<a id="ref1"></a>[1] Taylor, P. (2009). *Text-to-Speech Synthesis*. Cambridge University Press.
<a id="ref2"></a>[2] Oord, A. v. d., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., ... & Kavukcuoglu, K. (2016). WaveNet: A generative model for raw audio. *arXiv preprint arXiv:1609.03499*.
<a id="ref3"></a>[3] Shen, J., Pang, R., Weiss, R. J., Schuster, M., Jaitly, N., Yang, Z., ... & Wu, Y. (2018). Natural TTS synthesis by conditioning WaveNet on mel spectrogram predictions. *ICASSP*.
<a id="ref4"></a>[4] Kim, J., Kong, J., & Son, J. (2021). Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech. (VITS). *ICML*.
<a id="ref5"></a>[5] Casanova, E., Weber, J., Shulby, C., Gölge, E., et al. (2022). YourTTS: Towards zero-shot multi-speaker TTS and zero-shot voice conversion for everyone. *arXiv:2112.02418*.
<a id="ref6"></a>[6] Wang, C., Chen, S., Wu, Y., Zhang, Z., et al. (2023). Neural codec language models are zero-shot text-to-speech synthesizers. (VALL-E). *arXiv:2301.02111*.
<a id="ref7"></a>[7] Le, P., et al. (2023). Voicebox: Text-guided multilingual universal speech generation at scale. *Meta AI report*.
<a id="ref8"></a>[8] Anusuya, M. A., & Katti, S. K. (2024). Deep learning-based expressive speech synthesis: a systematic review. *EURASIP Journal on Audio, Speech, and Music Processing*.
<a id="ref9"></a>[9] IJCRT. (2025). A Comprehensive Review on Text-To-Speech (TTS). *International Journal of Creative Research Thoughts*, IJCRT2507281.
<a id="ref10"></a>[10] Cox, I. J., Miller, M. L., Bloom, J. A., Fridrich, J., & Kalker, T. (2008). *Digital Watermarking and Steganography*. Morgan Kaufmann.
<a id="ref11"></a>[11] Kirchenbauer, J., et al. (2023). Watermarking Language Models. *arXiv preprint arXiv:2302.06571*.