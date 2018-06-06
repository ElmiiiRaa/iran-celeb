<div dir="rtl">

# مجموعه ابزارها برای ایجاد مجموعه داده iran-celeb

## گام نخست: دانلود تصاویر
در گام نخست ابتدا 100 نتیجه نخست جست و جوی تصاویر گوگل را ذخیره کرده ایم.
بازیگران خود را از سایت iran-celeb.ir انتخاب کرده و آن‌ها را دانلود کنید

## گام دوم: حذف تصاویر نامرتبط
تمامی تصاویری که متعلق به بازیگر شما نیستند را پاک کنید.
اگر در یک تصویر چندچهره وجود داشت و در آن بین بازیگر مورد نظر نیز مشاهده می‌شد آن تصویر را پاک نمیکنیم.
همچنین تصاویر کودکی یا تصاویر با گریم‌های عجی بازیگر را پام نخواهیم کرد.

## گام سوم: انتخاب محدوده چهره
تصاویری که در این گام باقیمانده حتما حاوی چهره فرد مورد نظر است.

به تصاویری که تنها یک چهره در آن‌ها مشاهده می‌شود کاری نخواهیم داشت.

اما برای تصاویری که بیش از 1 چهره در آن‌ها وجود دارد باید محدوده چهره فرد مورد نظر مشخص شود.

بدین منظور از نرم افزار زیر استفاده میکنیم و آن را در حالت YOLO قرار میدهیم.


لطفا قسمت نصب
Windows + Anaconda

را بخوانید.
https://github.com/tzutalin/labelImg

اگر در یک تصویر چند چهره از فرد مربوط مشاهده می‌شد تمام محدوده‌های مربوط به آن فرد را انتخاب می‌کنیم.

## گام چهارم: بریدن محدوده‌های مشخص شده
در این گام تمام محدوده های مشخص شده crop میشوند.
بدین منظور اسکریپت 
[crop_face.py](https://github.com/Alireza-Akhavan/iran-celeb/blob/master/crop_face.py)
توسعه داده شده است.

برای استفاده از این اسکریپت ابتدا تمام فایل‌های دانلود شده را از زیپ خارج کرده و بدون تغییر در ساختار همه پوشه ها را در یک پوشه کپی کنید.
برای مثال  در اینجا همه پوشه ها از جمله کد 30 و کد 90 در پوشه ای با نام data کپی شده اند و ساختاری مشابه به ساختار زیر دارند.
</div>


  data/
  
      30/
        احمد مهرانفر
          1. 220px-ahmad_mehranfar_at_34th_fajr.jpg
          1. 220px-ahmad_mehranfar_at_34th_fajr.txt
          2. ahmadmehranfar-monafaezpour-10.jpg
          3. 0b3930c53394b80a728aa2a8e097046b.jpg
          ...

      90/
        بابک حميديان
          1. 220px-babak_hamidian_at_32th_fajr.jpg
          2. babak.jpg
          2. babak.txt
          ...
 
      ...

<div dir="rtl">
در ادامه آن را به صورت زیر فراخوانی میکنیم.
  در این مثال آدرس پوشه مبدا که ساختار آن در بالا تشریح شد 
 E:\face_data\data
  است و قصد داریم تصاویر خروجی بریده شده در پوشه
 E:\face_data\data_croped
  بروند:
 </div>
 
python crop_face.py E:\face_data\data E:\face_data\data_croped

<div dir="rtl">

## گام پنجم: تراز کردن چهره ها

ورودی این گام پوشه حاصل از خروجی گام چهارم است.
در این گام با الگوریتم تشخیص چهره MTCNN محدوده های چهره مشخص شده و پس از بریده شدن و افزودن یک margin مشخص به اندازه ی دلخواه برده می‌شوند.

بدین منظور اسکریپت 
[align_dataset_mtcnn.py](https://github.com/Alireza-Akhavan/iran-celeb/blob/master/mtcnn-align/align_dataset_mtcnn.py)
توسعه داده شده است.

مثالی از نحوه فراخوانی:

</div>

python align_dataset_mtcnn.py E:\face_data\data_croped E:\face_data\data_croped_mtcnnpy_182 --image_size 182 --margin 44
