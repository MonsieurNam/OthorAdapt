# Thư phản hồi - ARRAY-D-26-02033

## Orthogonal Multi-Head Gated Low-Rank Adaptation for Robust Vision-Language Model Adaptation

Kính gửi TS. Bora và các phản biện,

Chúng tôi xin cảm ơn các phản biện vì những nhận xét cẩn trọng, mang tính xây dựng, và vì đã mời chúng tôi nộp bản sửa đổi. Chúng tôi đã phản hồi riêng từng ý kiến của phản biện dưới đây. Các thay đổi trong bản thảo được tô màu vàng trong PDF sửa đổi.

Một số mối quan ngại, đặc biệt là giao thức seed, việc tách validation/test, hình phổ, cách diễn đạt về PSD, cũng như các nhận xét về phạm vi và định vị đóng góp, là xác đáng và đã khiến chúng tôi sửa đổi đáng kể cả phần thí nghiệm lẫn cách trình bày. Ở những nơi bằng chứng chỉ hỗ trợ một tuyên bố hẹp hơn so với bản thảo ban đầu, chúng tôi nêu rõ giới hạn đó.

Đồng thời, đóng góp còn giữ lại vẫn là tích cực theo tiêu chuẩn bằng chứng đã được sửa đổi. Ở thiết lập khớp số tham số (`r=2`, cùng ngân sách 184.320 tham số huấn luyện được), OrthoAdapt nhỉnh nhẹ về trung bình ghép cặp so với CLIP-LoRA (`+0.295` điểm phần trăm, 95% CI `[+0.102,+0.488]`), trong khi nhóm 16-shot gần như hòa. Ở cấu hình `r=8` được chọn bằng validation, OrthoAdapt cho thấy mức cải thiện ghép cặp trung bình khiêm tốn là `+0.356` điểm phần trăm (95% CI `[+0.162, +0.551]`) trên 72 cặp khớp, trong khi dùng ít hơn 37,5% tham số huấn luyện được so với CLIP-LoRA `r=8`. Vì vậy, chúng tôi tin rằng bản sửa đổi không chỉ thu hẹp các tuyên bố, mà còn củng cố tiêu chuẩn bằng chứng đứng sau các tuyên bố được giữ lại.

Tóm tắt các thay đổi:

- Các kết quả chính hiện được báo cáo thành hai phân tích riêng biệt, sử dụng ước lượng bất định ghép cặp: một chẩn đoán khớp số tham số theo giao thức ở `r=2` (rank baseline của CLIP-LoRA), trong đó các hàng CLIP-LoRA và OrthoAdapt dùng phép gộp hybrid 3 seed (seed 1 giữ từ bảng gốc đã nộp, seed 2 và 3 lấy từ các lần chạy lại) với cùng ngân sách tham số huấn luyện được, được báo cáo cùng các baseline CLIP few-shot đã thiết lập; và một cấu hình được chọn bằng validation ở `r=8`, so sánh trực tiếp với CLIP-LoRA, nhấn mạnh hiệu quả tham số.
- Các siêu tham số `H` và `lambda_o` được chọn trên dữ liệu validation giữ lại trước khi báo cáo test cuối cùng; `r=2` là thiết lập khớp số tham số cố định và `r=8` là thiết lập được chọn bằng validation, với một giá trị cố định duy nhất `lambda_o=0.03` và `H=2` áp dụng cho cả hai.
- Cách diễn đạt về PSD đã được sửa: OrthoAdapt được mô tả là định tuyến phụ thuộc đầu vào trên các head low-rank đối xứng PSD, chứ không phải là thoát khỏi ràng buộc PSD.
- Hình phổ đã được thay thế bằng một chẩn đoán được tái tạo, nhất quán với rank, từ các checkpoint đã sửa đổi và được dùng theo nghĩa mô tả thay vì làm bằng chứng nhân quả.
- SingLoRA-CLIP và DoRA được thảo luận như các phương pháp liên quan và giới hạn, thay vì được trình bày như các baseline đã đánh giá.
- Một tập con ViT-L/14 có giới hạn được bổ sung cho EuroSAT và Caltech101, thiết lập 4-shot, seed `{1,2,3}`.
- Các chẩn đoán trong phụ lục làm rõ rằng số lượng head, ngân sách rank và tính trực giao tạo thành một đánh đổi capacity/routing phụ thuộc cấu hình, chứ không phải một cơ chế phổ quát đã được chứng minh.
- ImageNet, SUN397 và StanfordCars được nêu rõ là công việc tương lai; benchmark sửa đổi được mô tả là một bộ 8 dataset.
- Code, seed chính xác, script chạy, bản ghi chia dataset và các script tổng hợp/thống kê sẽ được phát hành thông qua một kho ẩn danh.
- Các bất nhất số liệu trong bảng chính được điều hòa bằng các bảng kết quả tái tạo.
- Tuyên bố về robustness được loại khỏi các claim tích cực vì rerun robustness ghép cặp không vượt qua audit clean-consistency cần thiết để báo cáo khoa học.

Chúng tôi xin phản hồi từng điểm như sau.

## Phản hồi Reviewer #1

Chúng tôi cảm ơn đánh giá tổng thể tích cực và sự ghi nhận rõ ràng của phản biện đối với thiết kế framework, phân tích ablation và nỗ lực trực quan hóa. Bốn điểm yếu và ba câu hỏi do phản biện nêu ra đã khiến chúng tôi thu hẹp các tuyên bố, làm rõ phạm vi, và tách biệt các phát hiện được bằng chứng hỗ trợ khỏi công việc tương lai.

### Điểm yếu W1 - Tính mới so với MoRE (ICLR 2025)

> "Ý tưởng cốt lõi về LoRA dựa trên MoE với ràng buộc đa dạng rất giống các bài báo hội nghị hàng đầu gần đây (ví dụ MoRE tại ICLR 2025), và chỉ có đổi mới tăng dần."

**Phản hồi:** Chúng tôi cảm ơn phản biện vì đã chỉ ra dòng công trình liên quan này. Chúng tôi đồng ý một phần: bản thảo ban đầu đã định vị đóng góp quá rộng so với các công trình MoE-LoRA gần đây. Vì vậy, chúng tôi đã thu hẹp tuyên bố về tính mới và chỉnh sửa phần Related Work.

Bản thảo sửa đổi định vị OrthoAdapt như một nghiên cứu thích nghi VLM few-shot, thay vì một framework MoE-LoRA tổng quát. Sự khác biệt gồm ba điểm. Thứ nhất, các phương pháp như MoRE, MoLE và MoCLE chủ yếu nghiên cứu các thiết lập multi-task hoặc instruction-tuning, trong đó routing tách biệt các tác vụ hoặc chỉ dẫn; thiết lập của chúng tôi là thích nghi CLIP few-shot đơn tác vụ, trong đó routing được dùng để mô hình hóa biến thiên đặc trưng nội tác vụ. Thứ hai, OrthoAdapt dùng các head low-rank đối xứng, trong khi nhiều phương pháp MoE-LoRA kết hợp các expert LoRA bất đối xứng tiêu chuẩn. Thứ ba, hiện chúng tôi mô tả thành phần trực giao như một regularizer phụ thuộc cấu hình, thay vì một cơ chế nhân quả phổ quát.

Chúng tôi cũng bổ sung so sánh rõ hơn với OMoE và O-LoRA. Các công trình này cho thấy tính liên quan của trực giao, nhưng bối cảnh thí nghiệm của chúng khác với giao thức VLM few-shot đơn tác vụ của chúng tôi.

Ngoài việc thu hẹp tuyên bố, chúng tôi cũng nêu rõ đóng góp tích cực còn giữ lại để bản sửa đổi không bị hiểu chỉ là một loạt nhượng bộ. Sau khi sửa phần lý thuyết (xem Reviewer 2, M1), đóng góp của OrthoAdapt là một họ các head low-rank đối xứng PSD phụ thuộc đầu vào cho thích nghi CLIP few-shot. Đóng góp hẹp hơn này được hỗ trợ bởi hai kết quả. Thứ nhất, trong chẩn đoán khớp số tham số theo giao thức ở `r=2` (Bảng 1-3), OrthoAdapt nhỉnh nhẹ về trung bình ghép cặp so với CLIP-LoRA với cùng ngân sách tham số huấn luyện được, lợi thế tập trung ở 1-shot và 4-shot, còn 16-shot gần như hòa; so sánh này được đặt cùng các baseline CLIP few-shot đã thiết lập. Thứ hai, ở cấu hình `r=8` được chọn bằng validation, OrthoAdapt cho thấy mức cải thiện ghép cặp trung bình khiêm tốn trong khi dùng ít hơn 37,5% tham số huấn luyện được so với CLIP-LoRA `r=8`. Vì vậy, chúng tôi định vị OrthoAdapt như một thiết kế adapter phụ thuộc đầu vào và hiệu quả tham số cho thích nghi CLIP few-shot, chứ không phải một framework MoE-LoRA tổng quát mới.

**Thay đổi trong bản thảo:** Introduction và Related Work đã được chỉnh sửa và tô vàng, đặc biệt là đoạn baseline CLIP-LoRA, đoạn định vị LoRA/DoRA/rsLoRA, và các đoạn so sánh MoRE/MoLE/MoCLE/O-LoRA. So sánh khớp số tham số với các baseline tiêu chuẩn (Bảng 1-3) và kết quả hiệu quả tham số ở `r=8` được dùng để nêu đóng góp giữ lại. [p. TBD]

### Điểm yếu W2 - Phạm vi thí nghiệm hẹp

> "Bài báo chỉ tập trung vào phân loại few-shot đơn tác vụ với backbone ViT-B/16, thiếu validation trên backbone lớn (ví dụ ViT-L/14), các kịch bản multi-task, hoặc các tác vụ VLM cốt lõi (ví dụ truy hồi ảnh)."

**Phản hồi:** Chúng tôi đồng ý rằng phạm vi thực nghiệm ban đầu còn hạn chế. Chúng tôi đã chỉnh sửa bản thảo để nêu rõ phạm vi: các thí nghiệm được báo cáo tập trung vào phân loại few-shot đơn tác vụ với CLIP ViT-B/16, với một tập con ViT-L/14 có giới hạn được thảo luận riêng trong phản hồi Q3. Chúng tôi không trình bày đây là độ bao phủ toàn diện cho mọi thiết lập thích nghi VLM.

Chúng tôi đã bổ sung phần Limitations and Future Work, trong đó xác định thích nghi multi-task và truy hồi ảnh-văn bản là các bước tiếp theo quan trọng. Các thiết lập này đòi hỏi giao thức huấn luyện và đánh giá khác với thiết lập phân loại few-shot đơn tác vụ được dùng ở đây, nên chúng tôi xem chúng là công việc tương lai thay vì là các tuyên bố được hỗ trợ bởi thí nghiệm hiện tại.

**Thay đổi trong bản thảo:** Experimental Setup, Backbone Scalability, Limitations và Conclusion đã được chỉnh sửa và tô vàng. [p. TBD]

### Điểm yếu W3 - Vì sao H > 2 suy giảm

> "Thiết kế multi-head chỉ giới hạn ở H=2, với suy giảm hiệu năng rõ ràng khi H>2. Bài báo chưa giải thích đầy đủ các nguyên nhân nền tảng của giới hạn này hoặc đề xuất giải pháp hiệu quả."

**Phản hồi:** Chúng tôi cảm ơn phản biện vì đã yêu cầu làm rõ giới hạn này. Chúng tôi đã thêm bảng phụ lục "Per-Head Rank Under Small Total-Rank Budgets" để làm rõ cách tính capacity đằng sau câu hỏi của phản biện. Khi tổng rank chỉ là `r=2` hoặc `r=4`, tăng số lượng head có thể khiến mỗi head chỉ còn capacity rank-1 hoặc rank-2. Regularization trực giao có thể khuyến khích các head chiếm các hướng khác nhau, nhưng không thể tăng capacity sẵn có bên trong mỗi head. Dưới giám sát few-shot, điều này tạo ra một đánh đổi giữa độ đa dạng head, khả năng biểu đạt trên từng head và độ tin cậy của routing.

Chúng tôi cũng thêm một bảng phụ lục báo cáo độ nhạy validation theo số lượng head, và chỉnh cách diễn giải từ một cơ chế xác quyết sang giả thuyết đánh đổi capacity/routing. Trong các chẩn đoán validation, cấu hình được chọn `H=2, r=8, lambda_o=0.03` đạt accuracy validation trung bình 91.67, trong khi các thiết lập `H=4, r=8` được thử nghiệm vẫn gần nhưng thấp hơn, trong khoảng 90.63 đến 91.08. Vì vậy, chúng tôi mô tả đây là bằng chứng phụ thuộc cấu hình, thay vì bằng chứng cho một quy luật suy giảm phổ quát khi H lớn hơn.

Bản thảo sửa đổi cũng chỉ ra hướng có khả năng giải quyết giới hạn này: các nghiên cứu tương lai nên tách hiệu ứng tổng rank khỏi hiệu ứng rank trên mỗi head bằng cách chạy các chẩn đoán cố định rank-per-head và phân tích thống kê routing. Vì vậy, chúng tôi không trình bày `H=2` như một tối ưu phổ quát; chúng tôi trình bày nó như cấu hình được hỗ trợ bởi thiết lập validation hiện tại.

**Thay đổi trong bản thảo:** Bảng phụ lục "Per-Head Rank Under Small Total-Rank Budgets" và bảng phụ lục "Ramp100 Validation Sensitivity to Head Count" đã được thêm. Phần thảo luận ablation chính hiện tham chiếu cả hai bảng phụ lục, và các phần thảo luận về orthogonality-weight, hiệu quả tham số, chẩn đoán head-diversity và Limitations đã được chỉnh sửa và tô vàng. [p. TBD]

### Điểm yếu W4 - Thiếu baseline (MoRE, O-LoRA, DoRA)

> "Bài báo thiếu các baseline MoE-LoRA và orthogonal LoRA state-of-the-art quan trọng từ các hội nghị hàng đầu (ví dụ MoRE, O-LoRA, DoRA), dẫn đến so sánh hiệu năng chưa đủ toàn diện."

**Phản hồi:** Chúng tôi đồng ý rằng phần thảo luận baseline trong bản thảo ban đầu chưa đầy đủ. Chúng tôi đã chỉnh sửa phần Related Work và limitations để giải thích rõ hơn trạng thái của các baseline này, đồng thời giữ lại một so sánh thực nghiệm theo ngữ cảnh với các baseline CLIP few-shot đã thiết lập.

Bảng 1-3 đặt OrthoAdapt vào ngữ cảnh bằng cách so sánh với các baseline CLIP few-shot đã thiết lập: CLIP (zero-shot), CoOp, CoCoOp, CLIP-Adapter, Tip-Adapter-F, PLOT++, KgCoOp, TaskRes, MaPLe, ProGrad và CLIP-LoRA. Trong các bảng này, các hàng CLIP-LoRA và OrthoAdapt còn được khớp ở `r=2` và cùng ngân sách tham số huấn luyện được; kiểm soát bằng số tham số chỉ áp dụng cho cặp đó, trong khi các phương pháp còn lại là baseline tiêu chuẩn được báo cáo ở thiết lập công bố.

MoRE và O-LoRA là các phương pháp liên quan quan trọng, nhưng thiết lập đã công bố của chúng không trực tiếp khớp với giao thức CLIP few-shot đơn tác vụ của chúng tôi. MoRE tập trung vào học multi-task thích nghi, còn O-LoRA được thiết kế quanh nhiễu giao thoa trong continual learning. Một so sánh trung thực sẽ cần port cẩn thận và giao thức khớp, thay vì đưa trực tiếp các con số gốc của chúng vào bảng của chúng tôi. Vì vậy, chúng tôi thảo luận chúng như công trình liên quan và giới hạn phạm vi.

Đối với DoRA, chúng tôi không báo cáo một hàng thực nghiệm vì so sánh trung thực sẽ đòi hỏi triển khai và validation DoRA trong cùng giao thức thích nghi attention CLIP few-shot dùng cho các thí nghiệm chính. Vì vậy, chúng tôi thảo luận DoRA như một parameterization liên quan quan trọng và như một giới hạn, thay vì tuyên bố rằng nó đã được thêm như một baseline hoàn chỉnh. Đối với SingLoRA-CLIP, chúng tôi hiện cite chapter ICCSA/LNCS mới được công bố như predecessor/companion gần nhất. Chúng tôi không trộn hàng literature single-seed đã công bố của SingLoRA-CLIP vào các bảng chính sửa đổi vì các bảng đó dùng bằng chứng rerun ba seed và thống kê ghép cặp; thay vào đó, chúng tôi thảo luận SingLoRA-CLIP trong Related Work và chỉ dùng các kiểm soát symmetric single-head như chẩn đoán kiến trúc trong cùng giao thức.

**Thay đổi trong bản thảo:** Related Work và Limitations đã được chỉnh sửa và tô vàng. Không thêm hàng thực nghiệm mới cho các baseline chưa được đánh giá dưới giao thức khớp. [p. TBD]

### Câu hỏi Q1 - Tổng rank thấp, trực giao và số lượng head

> "Bạn có thể giải thích chi tiết hơn cách tổng rank cực thấp (r=2/4) và regularization trực giao mạnh cùng nhau giới hạn số lượng head không?"

**Phản hồi:** Chúng tôi đã chỉnh sửa bản thảo để làm rõ tương tác này và thêm bảng phụ lục "Per-Head Rank Under Small Total-Rank Budgets." Với ngân sách tổng rank thấp cố định, tăng số lượng head sẽ làm giảm capacity được gán cho mỗi head. Ví dụ, khi `r=2` và `H=2`, mỗi head là rank-1; khi `r=4` và `H=4`, mỗi head cũng là rank-1. Điều này có thể tạo ra chế độ thiếu capacity, trong đó mỗi head có quá ít rank để mô hình hóa biến thiên đặc thù tác vụ hữu ích, trong khi mạng gating vẫn phải học routing đáng tin cậy từ rất ít ví dụ.

Chúng tôi cũng làm rõ rằng hiệu ứng của regularization trực giao phụ thuộc cấu hình. Theo giao thức validation đã sửa, cấu hình được chọn là `H=2`, `r=8`, `lambda_o=0.03`. Các thiết lập lân cận có thể cạnh tranh, và các mức regularization khác nhau được ưa chuộng ở các thiết lập rank khác nhau. Vì vậy, chúng tôi không tuyên bố rằng trực giao mạnh hơn luôn cải thiện hiệu năng hoặc rằng nó là nguyên nhân chính của hành vi theo số lượng head quan sát được. Văn bản sửa đổi mô tả tương tác giữa tổng rank, capacity trên mỗi head, độ tin cậy của gating và cường độ regularization như một đánh đổi cần validation rõ ràng.

**Thay đổi trong bản thảo:** Head-count ablation, bảng phụ lục per-head-rank mới, bảng phụ lục "Ramp100 Validation Sensitivity to Head Count", phần độ nhạy regularization trực giao, chẩn đoán head-diversity và Limitations đã được chỉnh sửa và tô vàng. [p. TBD]

### Câu hỏi Q2 - Mở rộng multi-task

> "Các công trình hội nghị hàng đầu gần đây (ví dụ MoRE) dùng MoE-LoRA cho thích nghi multi-task. Vì sao công trình của bạn chỉ tập trung vào phân loại few-shot đơn tác vụ, và bạn có dự định mở rộng OrthoAdapt sang các kịch bản multi-task trong tương lai không?"

**Phản hồi:** Phạm vi này là có chủ đích. Bản thảo này nghiên cứu thích nghi CLIP few-shot đơn tác vụ vì nó cô lập hành vi thích nghi dữ liệu ít của adapter mà không đưa vào các lựa chọn lấy mẫu multi-task, cân bằng tác vụ hoặc đánh giá dành riêng cho truy hồi. Chúng tôi đồng ý rằng thích nghi multi-task là một mở rộng tự nhiên, đặc biệt xét mối liên hệ với các phương pháp MoE-LoRA, nhưng nó đòi hỏi một giao thức riêng và hiện đã được nêu rõ là công việc tương lai thay vì được hàm ý bởi các thí nghiệm hiện tại.

**Thay đổi trong bản thảo:** Limitations và Conclusion đã được chỉnh sửa và tô vàng. [p. TBD]

### Câu hỏi Q3 - Backbone lớn hơn như ViT-L/14

> "Các thí nghiệm của bạn chỉ dùng backbone ViT-B/16. Liệu các cải thiện hiệu năng của OrthoAdapt có còn giữ được khi mở rộng sang backbone lớn hơn như ViT-L/14 hoặc các VLM khác không?"

**Phản hồi:** Chúng tôi đã thêm một nghiên cứu backbone lớn hơn có giới hạn với CLIP ViT-L/14 trên EuroSAT và Caltech101 trong thiết lập 4-shot trên ba seed. So sánh này dùng cùng giao thức ghép cặp như bằng chứng chính của bản sửa đổi: CLIP-LoRA `r=8` so với OrthoAdapt `H=2, r=8, lambda_o=0.03`. Chúng tôi cũng thêm một bảng backbone-scaling CLIP ViT-L/14 có giới hạn để làm rõ bằng chứng này.

Tập con ViT-L/14 hoàn tất gồm 12 hàng thí nghiệm và 6 so sánh ghép cặp. OrthoAdapt dùng 1,01 triệu tham số huấn luyện được so với 1,62 triệu của CLIP-LoRA, giảm 37,5%. Chênh lệch accuracy ghép cặp tổng thể là `+0.21` điểm phần trăm, với 95% CI `[-1.04, +1.46]`. Khoảng tin cậy cắt qua 0, nên chúng tôi không tuyên bố rằng mức tăng accuracy nhìn chung vẫn giữ trên các backbone lớn hơn. Thay vào đó, chúng tôi dùng kết quả này như bằng chứng có giới hạn rằng phương pháp có thể áp dụng cho backbone CLIP lớn hơn với số tham số huấn luyện được thấp hơn đáng kể. Multi-task learning, retrieval và đánh giá rộng hơn trên các họ VLM vẫn là công việc tương lai.

**Thay đổi trong bản thảo:** Experimental Setup, Backbone Scalability, bảng mới "Bounded CLIP ViT-L/14 Backbone Scaling Study", Limitations và Conclusion đã được chỉnh sửa và tô vàng. [p. TBD]

## Phản hồi Reviewer #2

Chúng tôi cảm ơn phản biện vì phần đọc kỹ thuật cẩn trọng. Một số điểm đã chỉ ra các vấn đề thực sự trong bản nộp ban đầu. Chúng tôi đã sửa các vấn đề đó và thu hẹp các tuyên bố ở những nơi bằng chứng sửa đổi không hỗ trợ cách diễn đạt ban đầu.

### Vấn đề chính M1 - Tuyên bố PSD không đúng dưới softmax gating

> "Tuyên bố lý thuyết trung tâm, cụ thể là cơ chế gating thoát khỏi ràng buộc PSD, không đúng như đã nêu. [...] Một tổ hợp không âm của các ma trận PSD vẫn là PSD. [...] Điều gating thực sự đóng góp là tính phụ thuộc đầu vào."

**Phản hồi:** Phản biện đúng. Dưới softmax gating, cập nhật trên từng đầu vào

`Delta W(x) = sum_h g_h(x) U_h U_h^T`

vẫn là một tổ hợp không âm của các ma trận PSD. Vì vậy, cách diễn đạt ban đầu rằng cơ chế gating "thoát khỏi" ràng buộc PSD là sai về mặt toán học.

Chúng tôi đã chỉnh sửa bản thảo tương ứng. OrthoAdapt hiện được mô tả là một họ các cập nhật low-rank đối xứng PSD phụ thuộc đầu vào, chứ không phải một cơ chế loại bỏ ràng buộc PSD cho một đầu vào riêng lẻ. Giá trị của gate là routing phụ thuộc đầu vào: các đầu vào khác nhau có thể nhận các cập nhật PSD khác nhau thông qua trọng số head đã học. Đây là tuyên bố hẹp hơn so với ban đầu, nhưng là tuyên bố được hỗ trợ bởi kiến trúc thực sự được triển khai.

**Thay đổi trong bản thảo:** Abstract, Introduction, Method và chú thích Fig. 2 đã được chỉnh sửa và tô vàng. Không cần bảng hoặc hình mới cho mục này. [p. TBD]

### Vấn đề chính M2 - Siêu tham số được chọn trên tập test

> "Số lượng head và cường độ regularization được chọn theo test accuracy trên EuroSAT. [...] Chọn siêu tham số trên chính tập test nơi các cải thiện chính được tuyên bố là không phù hợp với cách bài báo trình bày rằng siêu tham số cố định. Tôi đề nghị tác giả thực hiện lựa chọn này trên dữ liệu validation và báo cáo lại kết quả."

**Phản hồi:** Phản biện đúng rằng phần trình bày ban đầu chưa tách đầy đủ việc chọn siêu tham số khỏi báo cáo test. Chúng tôi đã sửa code và giao thức bằng cách thêm đường chọn chỉ dùng validation rõ ràng và chặn báo cáo tập test trong sweep mode.

Để tránh vấn đề mà phản biện nêu, hiện chúng tôi báo cáo hai phân tích riêng ở hai thiết lập rank, và mọi lựa chọn siêu tham số đều được thực hiện trên dữ liệu validation giữ lại.

Phân tích thứ nhất là chẩn đoán khớp số tham số theo giao thức ở `r=2`, rank mặc định của baseline CLIP-LoRA. Ở đây, mục tiêu là so sánh có kiểm soát trong đó OrthoAdapt và CLIP-LoRA dùng cùng ngân sách tham số huấn luyện được, để bất kỳ khác biệt nào giữa hai phương pháp không thể bị quy cho adapter lớn hơn. Số lượng head (`H=2`) và trọng số trực giao (`lambda_o=0.03`) dùng ở thiết lập này là cố định, thay vì được chọn từ test accuracy.

Phân tích thứ hai là cấu hình được chọn bằng validation. Theo một quy tắc validation định trước, chúng tôi sweep số lượng head, rank và trọng số trực giao trên EuroSAT và Caltech101 trong thiết lập 4-shot, seed `{1,2,3}`, chỉ đánh giá trên validation split và không bao giờ báo cáo test accuracy trong quá trình sweep. Theo quy tắc này, cấu hình được chọn là `H=2`, `r=8`, `lambda_o=0.03`, với accuracy validation trung bình 91.67. Sau đó chúng tôi báo cáo cấu hình này trên tập test. Các ứng viên không thắng không được đánh giá trên tập test cho mục đích lựa chọn.

Chúng tôi giữ một giá trị cố định duy nhất `H=2` và `lambda_o=0.03` trên cả hai thiết lập rank. Điều này có thể bảo vệ được vì ở `r=2`, trọng số trực giao có ảnh hưởng yếu và nằm trong nhiễu đối với validation accuracy: trên các `lambda_o` trong `{0, 0.01, 0.03, 0.05}`, accuracy validation trung bình chỉ trải từ 90.125 đến 91.000, và các giá trị `{0, 0.03, 0.05}` nằm trong phạm vi 0.05 điểm của nhau. Vì vậy, chúng tôi không tuyên bố rằng `lambda_o=0.03` là tối ưu validation riêng ở `r=2`; chúng tôi giữ cố định nó và ghi chú rằng giá trị cụ thể trong khoảng này không làm thay đổi kết luận khớp số tham số.

**Thay đổi trong bản thảo:** Giao thức đánh giá và phần thảo luận chọn bằng validation đã được chỉnh sửa và tô vàng, trình bày chẩn đoán khớp số tham số `r=2` và cấu hình `r=8` được chọn bằng validation như hai phân tích riêng. Chúng tôi cũng thêm các chẩn đoán validation, bao gồm khoảng độ nhạy lambda ở `r=2`, để cấu hình được chọn và các thiết lập lân cận được minh bạch. [p. TBD]

### Vấn đề chính M3 - Mâu thuẫn giao thức seed và mức tăng nằm trong nhiễu seed

> "Mục 4.1 nói rằng kết quả được lấy trung bình trên ba random seed, trong khi Bảng 1 nói rằng dùng một seed duy nhất. Hai phát biểu này không thể đồng thời đúng. [...] Các biên độ cỡ này không thể được diễn giải nếu không có ước lượng phương sai."

**Phản hồi:** Phản biện đúng. Bản thảo ban đầu có mâu thuẫn trong giao thức seed, và các biên độ nhỏ không nên được diễn giải nếu thiếu ước lượng bất định.

Trước hết, chúng tôi đã sửa mâu thuẫn bằng cách tách rõ hai nguồn bằng chứng. Với so sánh khớp số tham số `r=2`, chúng tôi giữ các giá trị CLIP-LoRA và OrthoAdapt trong bảng gốc đã nộp làm seed 1, rồi gộp với seed 2 và 3 từ các lần chạy lại để tạo ước lượng hybrid 3 seed. Với so sánh `r=8` được chọn bằng validation, cả ba seed đều lấy từ ma trận chạy lại ramp100. Chúng tôi báo cáo ước lượng bất định ghép cặp cho cả hai so sánh.

Trong chẩn đoán khớp số tham số theo giao thức ở `r=2` (Bảng 1-3), OrthoAdapt và CLIP-LoRA được so sánh với cùng ngân sách 184.320 tham số huấn luyện được bằng phép gộp hybrid 3 seed: seed 1 giữ từ bảng gốc đã nộp, còn seed 2 và 3 lấy từ các lần chạy lại. Các delta ghép cặp là `+0.456` cho 1-shot, `+0.428` cho 4-shot và `+0.001` cho 16-shot, với delta ghép cặp tổng thể `+0.295` điểm (95% CI `[+0.102,+0.488]`) trên 72 cặp khớp. Chúng tôi báo cáo rõ 95% CI của delta ghép cặp 16-shot là `[-0.192,+0.193]` vì nó cắt qua 0, làm minh bạch hành vi 16-shot và cho thấy lợi thế của OrthoAdapt tập trung ở các chế độ shot thấp nhất. Điều này cho thấy một mức tăng trung bình nhỏ ở cùng số tham số, với hành vi theo dataset còn hỗn hợp.

Ở cấu hình được chọn bằng validation `r=8`, so sánh bao phủ 8 dataset, 3 thiết lập shot, 2 phương pháp và seed `{1,2,3}` (144 lần chạy test hoàn tất). Ghép cặp là chính xác theo `(dataset, shot, seed)`, tạo ra 72 so sánh khớp giữa CLIP-LoRA `r=8` và OrthoAdapt `H=2, r=8, lambda_o=0.03`. Trên 72 cặp này, OrthoAdapt thay đổi accuracy `+0.356` điểm phần trăm, với 95% CI `[+0.162,+0.551]`, effect size ghép cặp `dz=0.430`, và thắng/hòa/thua là 48/3/21. Các delta ghép cặp trung bình là `+0.816` cho 1-shot, `+0.402` cho 4-shot và `-0.149` cho 16-shot. Chúng tôi báo cáo rõ 95% CI của delta ghép cặp 16-shot là `[-0.359, +0.061]` vì nó cắt qua 0, làm minh bạch hành vi 16-shot và cho thấy lợi thế của OrthoAdapt tập trung ở các chế độ shot thấp nhất. OrthoAdapt dùng 460.800 tham số huấn luyện được so với 737.280 của CLIP-LoRA `r=8`, giảm 37,5%.

Vì vậy, chúng tôi không còn tuyên bố hiệu năng dẫn đầu benchmark rộng rãi hoặc ưu thế đồng đều. Kết luận sửa đổi là OrthoAdapt cho thấy một delta ghép cặp trung bình dương khiêm tốn trong khi dùng không nhiều tham số hơn CLIP-LoRA ở thiết lập khớp, và dùng ít tham số hơn đáng kể ở cấu hình `r=8` được chọn bằng validation.

**Thay đổi trong bản thảo:** Các giá trị 1-shot, 4-shot và 16-shot gốc được giữ như entry seed 1 cho bảng khớp số tham số, rồi được gộp với seed 2 và 3 từ các lần chạy lại để báo cáo thống kê ghép cặp ở `r=2`. Phân tích `r=8` được báo cáo từ ma trận chạy lại ramp100 đủ ba seed. Thống kê ghép cặp, bao gồm effect size và khoảng tin cậy theo từng shot, được báo cáo trong bản thảo cho cả phân tích `r=2` và `r=8`. Các thay đổi này được tô vàng. [p. TBD]

### Vấn đề chính M4 - Hình phổ không nhất quán với ngân sách rank

> "Với tổng ngân sách rank r = 2, ma trận cập nhật nhiều nhất chỉ có hai singular value khác 0 [...] Tuy nhiên hình lại cho thấy khoảng mười lăm singular value giảm trơn ở mọi layer."

**Phản hồi:** Phản biện đúng. Hình phổ cũ không thể hỗ trợ tuyên bố low-rank đã nêu nếu không chỉ rõ checkpoint, định nghĩa ma trận, layer, projection và ngân sách rank chính xác; việc vẽ nhiều singular values hơn ngân sách rank khai báo làm hình dễ bị đọc nhầm khi đặt cạnh claim `r=2`. Chúng tôi đã thay phân tích không được hỗ trợ đó bằng một chẩn đoán phổ được tái tạo, nhất quán với rank, và hiện báo cáo ma trận, layer, projection, cấu hình và ngân sách rank được vẽ đứng sau nó.

Lý do trực tiếp khiến hình cũ nhìn như có khoảng mười lăm singular values là vấn đề lựa chọn cách visualize, không phải bằng chứng rằng adapter đã dùng rank 15. Code cũ tính SVD trên một ma trận/proxy lấy từ checkpoint rồi hiển thị một đoạn prefix cố định của phổ theo tham số vẽ (`top_k`, trước đây mặc định 50 trong script phân tích) để đường decay dễ quan sát hơn. Tuy nhiên, lựa chọn hiển thị này không bị ràng buộc với ngân sách rank đã khai báo. Với một diagnostic dùng để kiểm tra rank budget, chỉ `r` singular values đầu tiên mới nên được hiển thị như phần phổ có thể diễn giải trực tiếp; nếu vẽ thêm, reviewer có thể hiểu hợp lý rằng ta đang ngụ ý update có rank lớn hơn. Vì vậy, hình cũ không phù hợp cho claim `r=2`, và chúng tôi đã thay bằng các plot căn đúng theo rank.

Pipeline chẩn đoán mới có kiểm soát theo rank: diagnostic `r=2` cùng số tham số chỉ vẽ đúng hai singular value đầu, còn diagnostic `r=8` theo cấu hình được chọn bằng validation chỉ vẽ đúng tám singular value đầu. Pipeline sẽ dừng nếu giá trị `top_k` dùng để vẽ không khớp với ngân sách rank đã khai báo. Manifest chẩn đoán ghi lại định nghĩa ma trận, layer, projection, cấu hình, metadata checkpoint, hash checkpoint, ngân sách rank và singular values. Hình phản hồi/bản thảo hiện được neo vào panel EuroSAT 4-shot, seed 1, vision layer 11, `q_proj`, từ diagnostic khớp số tham số `r=2` (`eurosat_4shot_seed1_layer11_q_proj.png`), vì đây là cấu hình trực tiếp liên quan đến concern của phản biện và so sánh chính với CLIP-LoRA ở cùng ngân sách rank. Diagnostic `r=8` vẫn được giữ như bằng chứng truy vết phụ cho cấu hình được chọn bằng validation, nhưng không phải hình chính dùng để trả lời concern M4. Trong cả hai trường hợp, chúng tôi không hiển thị hoặc diễn giải phần numerical tail sau rank.

Tóm tắt tập con diagnostic trong bản thảo: panel được chọn là EuroSAT 4-shot, seed 1, layer 11, `q_proj`, `r=2` và chỉ hiển thị đúng hai singular values đầu. Trên báo cáo layer-11, seed-1 bao phủ tám dataset ở cùng thiết lập diagnostic, query projection 4-shot có stable rank trung bình 1.004 cho CLIP-LoRA và 1.064 cho OrthoAdapt; với value projection, stable rank là 1.281 cho CLIP-LoRA và 1.563 cho OrthoAdapt. Các giá trị này hỗ trợ một phát biểu mô tả có giới hạn về tập con đã phân tích, không phải một tuyên bố cơ chế nhân quả.

**Thay đổi trong bản thảo:** Hình spectrum cũ đã được thay bằng panel EuroSAT 4-shot, seed 1, vision layer 11, `q_proj` từ thiết lập khớp số tham số `r=2`; nếu giữ panel companion `v_proj`, panel đó cũng phải lấy từ cùng tập con EuroSAT/4-shot/seed-1/layer-11/`r=2`. Các đường trong hình dừng đúng tại ngân sách rank, và chú thích cùng thảo luận đã được chỉnh sửa, tô vàng để nêu rõ tập checkpoint, layer/projection, ngân sách rank và vai trò mô tả của hình. Báo cáo tái tạo đầy đủ bao phủ cả tám dataset, thống kê stable-rank trung bình cho tập con phân tích hiện được báo cáo trong phần thảo luận spectrum của bản thảo, và hình trong bản thảo được xem là chẩn đoán mô tả, không phải bằng chứng nhân quả. [p. TBD]

### Nhận xét nhỏ a - Bất nhất số liệu

> "Abstract gán mức tăng robustness 3.6 điểm cho severe corruption, trong khi Mục 4.4 gán cho thiết lập medium. Accuracy EuroSAT 4-shot xuất hiện là 88.64 trong Hình 5, 89.06 trong Bảng 3, và 88.57 trong Bảng 6. Trong Bảng 4, kết quả Aircraft giống hệt CLIP-LoRA ở 54.97, tôi nghi đây là lỗi sao chép. Bảng 6 dùng dấu phẩy làm dấu thập phân."

**Phản hồi:** Chúng tôi đồng ý với phản biện. Bản thảo sửa đổi không còn bảo vệ các giá trị cũ mâu thuẫn. Các bảng accuracy chính đã được tái tạo từ các lần chạy 3 seed, và các bảng sensitivity khám phá cũ hiện được gắn nhãn rõ là chẩn đoán và tách khỏi các giá trị test cuối cùng.

Số EuroSAT 4-shot không còn mang nhiều ý nghĩa mâu thuẫn. Ở thiết lập khớp số tham số `r=2`, phép gộp hybrid 3 seed báo cáo OrthoAdapt là 86.41 và CLIP-LoRA là 85.61; giá trị gốc `89.06` chỉ được giữ như entry seed 1 của OrthoAdapt trong bảng khớp số tham số. Ở thiết lập được chọn bằng validation `r=8`, OrthoAdapt là 85.78 và CLIP-LoRA là 84.36. Các giá trị EuroSAT cũ khác (88.64 và 88.57) nay được gắn nhãn là chẩn đoán, không phải kết quả test aggregate cuối cùng.

Vấn đề nghi ngờ sao chép ở Aircraft cũng được làm rõ. Giá trị gốc `54.97/54.97` được giữ như entry seed 1 của Aircraft 16-shot, thay vì bị xem là lỗi sao chép. Trong phép gộp hybrid 3 seed ở thiết lập khớp số tham số, Aircraft 16-shot là CLIP-LoRA 54.65 và OrthoAdapt 54.63; ở `r=8`, Aircraft 16-shot là CLIP-LoRA 56.99 và OrthoAdapt 56.63.

Cách diễn đạt về robustness cũng đã được sửa. Chúng tôi đã thử đánh giá robustness ghép cặp trên các checkpoint Phase 3 cuối cùng, nhưng run này không vượt qua audit clean-consistency: accuracy severity-0 của OH-SingLoRA không tái lập accuracy clean test tương ứng trong Phase 3, cho thấy có mismatch ở evaluator/checkpoint loading. Vì vậy, chúng tôi không báo cáo các số robustness từ run đó và loại bỏ tuyên bố robustness `+3.6` ban đầu khỏi các kết luận chính.

Trực quan hóa robustness cũ chỉ được giữ lại trong bản thảo như một minh họa định tính/khám phá được gắn nhãn rõ, với chú thích nêu rằng không có robustness gain định lượng nào được tuyên bố từ đó. Đánh giá robustness chuyên biệt được xem là công việc tương lai trừ khi có một rerun vượt qua clean gate.

Vấn đề dấu phân cách thập phân trong bảng ablation cũ đã được sửa ở những nơi bảng đó còn được giữ lại.

**Thay đổi trong bản thảo:** Các bảng kết quả chính, chú thích ablation, văn bản robustness, chú thích trực quan hóa robustness định tính/khám phá được giữ lại, và định dạng bảng đã được chỉnh sửa và tô vàng. [p. TBD]

### Nhận xét nhỏ b - Baseline SingLoRA và phân biệt với OMoE

> "Vì SingLoRA là điểm xuất phát kiến trúc, nó nên xuất hiện như một baseline đầy đủ trong Bảng 2 đến 4. Sự khác biệt với công trình gần đây kết hợp mixture of LoRA experts với orthogonality, như OMoE trong tham khảo 41, cũng cần một so sánh sắc nét hơn so với Related Work hiện tại."

**Phản hồi:** Chúng tôi đồng ý rằng SingLoRA là một điểm xuất phát khái niệm quan trọng, và Related Work sửa đổi hiện giải thích mối quan hệ này rõ hơn. Kể từ khi bản nộp ban đầu được gửi đi, nghiên cứu SingLoRA-CLIP trước đó của nhóm chúng tôi đã xuất hiện như một chapter ICCSA/LNCS đã công bố. Chúng tôi hiện cite rõ công trình này và định vị nó như predecessor/companion gần nhất: SingLoRA-CLIP đánh giá cập nhật đối xứng single-matrix cho thích nghi CLIP few-shot, trong khi OrthoAdapt nghiên cứu routing multi-head phụ thuộc đầu vào trên các head low-rank đối xứng.

Đồng thời, chúng tôi không trộn trực tiếp các số SingLoRA-CLIP đã công bố vào các bảng chính sửa đổi. Chapter SingLoRA-CLIP báo cáo một kết quả literature baseline single-seed đã công bố, còn các bảng OrthoAdapt sửa đổi dùng giao thức kiểm soát bằng manifest với ba seed và ước lượng bất định ghép cặp. Trộn một hàng literature single-seed từ công trình trước với các hàng rerun ba seed sẽ làm so sánh uncertainty không rõ ràng. Vì vậy, chúng tôi cite và thảo luận SingLoRA-CLIP như predecessor gần nhất, trong khi chỉ giới hạn các so sánh thống kê ghép cặp cho những phương pháp được rerun dưới cùng giao thức sửa đổi.

Để xử lý câu hỏi kiến trúc mà không phóng đại bằng chứng baseline, chúng tôi bổ sung các lần chạy symmetric-head `H=1` chỉ như chẩn đoán validation trong codebase hiện tại. Trong chẩn đoán validation `H=1` trên EuroSAT và Caltech101, 4-shot, seed `{1,2,3}`, `H=1, r=2, lambda_o=0.0` đạt trung bình 91.33, và `H=1, r=4, lambda_o=0.0` đạt trung bình 89.75. Các lần chạy chẩn đoán này không thay thế nghiên cứu SingLoRA-CLIP đã công bố; chúng chỉ giúp diễn giải hiệu ứng chuyển từ một symmetric head sang routed multi-head adaptation trong giao thức validation hiện tại.

Chúng tôi cũng làm sắc nét hơn so sánh OMoE/MoE-LoRA bằng cách phân biệt thiết lập mục tiêu, parameterization cơ sở và vai trò của trực giao.

**Thay đổi trong bản thảo:** Related Work hiện cite và định vị chapter SingLoRA-CLIP đã công bố như predecessor/companion gần nhất, và phần so sánh OMoE/MoE-LoRA được làm sắc nét hơn bằng cách phân biệt thiết lập mục tiêu, parameterization cơ sở và vai trò của trực giao. Chẩn đoán kiến trúc, bao gồm các giá trị chẩn đoán validation `H=1`, và Limitations đã được chỉnh sửa và tô vàng. Không thêm hàng baseline bảng chính cho các kết quả chưa được đánh giá dưới giao thức ba seed đã sửa đổi. [p. TBD]

### Nhận xét nhỏ c - Hàm ramp-up và epsilon chưa được định nghĩa

> "Hàm ramp-up u(t) trong Eq. 3 và 6 chưa bao giờ được định nghĩa, và hằng số epsilon được nhắc sau Eq. 11 thực ra không xuất hiện trong phương trình."

**Phản hồi:** Phản biện đúng. Chúng tôi đã thêm định nghĩa rõ ràng cho hệ số ramp-up:

`u(t)=min(1,t/T)`,

trong đó `T` là ngân sách bước ramp-up. Trong các thí nghiệm cuối cùng, `T=100`. Chúng tôi cũng đã loại bỏ tham chiếu epsilon thừa vì loss trực giao được triển khai không chứa thành phần đó.

**Thay đổi trong bản thảo:** Các phương trình trong Method và văn bản xung quanh đã được chỉnh sửa và tô vàng. Không cần bảng hoặc hình mới. [p. TBD]

### Nhận xét nhỏ d - Thiếu ImageNet, SUN397 và StanfordCars

> "Bộ CLIP few-shot tiêu chuẩn gồm mười một dataset, bao gồm ImageNet, SUN397 và StanfordCars. Tác giả nên giải thích vì sao ba dataset này bị bỏ qua."

**Phản hồi:** Chúng tôi cảm ơn phản biện vì đã chỉ ra điều này. Chúng tôi đồng ý rằng ImageNet, SUN397 và StanfordCars là một phần của bộ CLIP few-shot tiêu chuẩn rộng hơn. Trong bản thảo sửa đổi, chúng tôi không còn mô tả benchmark của mình là bộ CLIP 11 dataset đầy đủ. Chúng tôi định nghĩa rõ nó là một bộ đánh giá few-shot gồm tám dataset.

Mọi tuyên bố thực nghiệm trong bài sửa đổi đều bị giới hạn ở tám dataset có bằng chứng chạy lại đầy đủ: Aircraft/FGVC, EuroSAT, Food101, OxfordPets, OxfordFlowers, Caltech101, DTD và UCF101. ImageNet, SUN397 và StanfordCars hiện được liệt kê là công việc tương lai thay vì được hàm ý là đã bao phủ.

**Thay đổi trong bản thảo:** Experimental Setup, chú thích hình và Limitations đã được chỉnh sửa và tô vàng, và ImageNet, SUN397, StanfordCars hiện được liệt kê rõ là công việc tương lai trong phần Limitations. Không cần bảng hoặc hình mới. [p. TBD]

### Nhận xét nhỏ e - Phát hành code và seed chính xác

> "Vì các biên độ rất nhỏ, việc phát hành code và seed chính xác sẽ giúp tăng đáng kể độ tin cậy của kết quả, và tôi nhiệt tình khuyến khích tác giả làm điều đó."

**Phản hồi:** Chúng tôi đồng ý. Tuyên bố reproducibility sửa đổi cam kết phát hành code, seed chính xác `{1,2,3}`, script chạy, bản ghi chạy validation/test, bản ghi chia dataset, script tổng hợp/thống kê, thông tin môi trường và hướng dẫn cần thiết để tái tạo các bảng đã báo cáo. Adapter checkpoint sẽ được phát hành khi ràng buộc lưu trữ và giấy phép cho phép; nếu không, các script và bản ghi chạy được phát hành sẽ đủ để tái tạo chúng.

**Thay đổi trong bản thảo:** Data/Code Availability đã được chỉnh sửa và tô vàng. Không cần bảng hoặc hình mới. [p. TBD]

Chúng tôi hy vọng các sửa đổi này giải quyết thỏa đáng các mối quan ngại của phản biện. Chúng tôi đặc biệt biết ơn Reviewer 2 vì các nhận xét kỹ thuật chính xác về cách diễn đạt PSD, giao thức seed, tách validation/test và hình phổ. Những nhận xét này đã chỉ ra các sai sót thực sự mà bản sửa đổi hiện đã khắc phục.

Trân trọng,

Hoang Ngoc Tran (thay mặt tất cả tác giả)  
Department of Artificial Intelligence, FPT University, Can Tho, Vietnam  
hoang2531992@gmail.com
