# اصلاح patch روی شاخهٔ اصلی iTransformer

پچ قبلی اشتباهاً متدهای `load_checkpoint` و `rank_analysis` و پرچم‌های `induced_attention` / `warm_start_checkpoint` را نیز از شاخهٔ دیگری به شاخهٔ اصلی منتقل کرده بود. این بسته فقط RSL دارد.

در branch آزمایشی خودت، ابتدا وضعیت را ببین (`git status --short`). سپس **فقط چهار فایل داخل این بسته** را در مسیرهای هم‌نام پروژه جایگزین کن: `run.py`، `experiments/exp_long_term_forecasting.py`، `utils/rsl.py`، `scripts/traffic.bat`. فایل `data_provider/data_factory.py` فعلی را تغییر نده؛ در patch تو تغییری برای آن دیده نمی‌شود.

با `git diff -- run.py experiments/exp_long_term_forecasting.py` تأیید کن که تنها آرگومان‌های RSL، validation آن‌ها، import، ساخت PCA در train و افزودن loss در دو شاخهٔ AMP/معمولی دیده شوند. همچنین `git diff --check` را اجرا کن.

در `train()`، کد اصلی iTransformer بعد از early stopping بهترین checkpoint **همان اجرای تازه** را با `torch.load(best_model_path)` بارگذاری می‌کند؛ آن را حفظ کرده‌ایم. فایل `checkpoint.pth` ارسالی برای این آزمایش استفاده نمی‌شود. دو فرایند فایل batch از صفر train می‌شوند؛ ابتدا alpha=0 و سپس alpha=0.5. داده باید در `dataset/traffic/traffic.csv` باشد و محیط Python/PyTorch خودت فعال باشد.
