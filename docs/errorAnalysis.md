
PII Detection Model Error Analysis
The main areas of analysis are as follows.
•	False Negative: Cases where PII is present but the model fails to detect it
•	False Positive: Cases where the model detects non-PII as PII
•	Type Confusion: Cases where the model detects the location of PII but classifies it as the wrong type
•	Boundary Error: Cases where the model detects only part of the PII rather than its full span
•	Unusual Format: Cases where the model misses formats that do not frequently appear in the training data
________________________________________
2. Summary of Manual Test Results
There are 10 representative examples.
Among them, the following results were observed in 9 examples that could be clearly evaluated.
Result	Count
Correct detection	4
False Negative	3
Type Confusion	1
False Positive	1
The main errors occurred with phone numbers, short usernames, and certain email formats.
________________________________________
3. Analysis by Error Type
3.1 False Negative for Phone Number Formats
Input:
Please call me at 415-555-1234 tomorrow.
Expected Result:
PHONE_NUM = 415-555-1234
Model Result:
No PII detected.
Error Type:
False Negative
Possible Cause:
Phone numbers in the xxx-xxx-xxxx format commonly used in the United States may not have been sufficiently represented in the training data.
The model failed to detect the number despite the presence of the phone number context, “call me at.” Therefore, its ability to generalize to the phone number format itself may be limited.
________________________________________
3.2 Type Confusion Between Phone Numbers and Addresses
Input:
My phone number is 408-555-4321.
Expected Result:
PHONE_NUM = 408-555-4321.
Model Result:
STREET_ADDRESS = 408-555-4321.
Error Type:
Type Confusion
PHONE_NUM -> STREET_ADDRESS
Possible Cause:
The model recognized that the string was PII but classified it as the wrong type of PII.
This case demonstrates an error in the entity type classification process rather than a failure of PII detection itself.
________________________________________
3.3 False Negative for Short Usernames
Input:
My username is dk123.
Expected Result:
USERNAME = dk123.
Model Result:
No PII detected.
Error Type:
False Negative
Possible Cause:
Short, common combinations of English letters and numbers such as dk123 are difficult to distinguish from ordinary words or codes based on their form.
________________________________________
3.4 False Negative for Email Formats
Input:
You can email me at test@example.net.
Expected Result:
EMAIL = test@example.net.
Model Result:
No PII detected.
Error Type:
False Negative
Possible Cause:
The model failed to detect the email address despite the presence of strong contextual information in the sentence, “email me at.”
Generalization to domains such as .net or simple formats such as test@example.net may be insufficient.
________________________________________
3.5 Confusion Between Personal URLs and Non-Personal URLs
Input:
I read the news every day on https://www.cnn.com.
Expected Result:
Non-personal URL
Model Result:
URL_PERSONAL = https://www.cnn.com.
Error Type:
False Positive
Possible Cause:
Although the model recognizes URL formats well, its ability to distinguish whether a URL is a personal URL or a general public website may be limited.
In other words, the model may not fully distinguish between:
URL format
and
Personal URL
in some cases.
This case demonstrates the need to make judgments based not only on the structure of the URL but also on its surrounding context and meaning.
________________________________________
4. Correct Detection Cases
In addition to errors, cases that the model handled well were also identified.
Name
Input:
My name is Maria Gonzalez.
Result:
NAME_STUDENT = Maria Gonzalez.
The model correctly detected a person’s name in the context of “My name is.”
Address
Input:
I live at 1234 West Pine Street Apt. 5.
Result:
STREET_ADDRESS = 1234 West Pine Street Apt. 5.
The model detected the entire long address with the correct boundaries.
Personal URL
Input:
My personal website is https://www.johnsmith.dev/profile.
Result:
URL_PERSONAL = https://www.johnsmith.dev/profile.
The model correctly detected the URL in the clear context of a personal website.
Sentence Without PII
Input:
Deep learning models can learn useful representations from large datasets.
Result:
No PII detected.
The model did not generate unnecessary PII in an ordinary sentence without PII.
________________________________________
5. Context Dependence
The following sentence was additionally tested.
John Smith met Sarah Johnson yesterday.
Model Result:
No PII detected.
In contrast, the model correctly detected the name in the following sentence.
My name is Maria Gonzalez.
This result shows that the model does not simply classify every name that appears to be a person’s name as PII, but is influenced by the surrounding context.
This may help reduce False Positives in which ordinary personal names are incorrectly detected as PII, but it may also cause the model to miss names that constitute actual personal information when the context is unclear.
________________________________________
6. Representative Error Cases
The following are representative error cases to use in the report and presentation materials.
Example	Expected	Prediction	Error Type	Possible Cause
415-555-1234	PHONE_NUM	None	False Negative	Unusual U.S. phone format
408-555-4321	PHONE_NUM	STREET_ADDRESS	Type Confusion	Phone/address pattern confusion
dk123	USERNAME	None	False Negative	Short username
test@example.net	EMAIL	None	False Negative	Email/domain generalization
https://www.cnn.com	Non-personal URL	URL_PERSONAL	False Positive	Personal/non-personal URL confusion
The following examples can also be used as comparison cases to demonstrate correct operation.
Example	Prediction	Result
Maria Gonzalez	NAME_STUDENT	Correct
1234 West Pine Street Apt. 5.	STREET_ADDRESS	Correct
https://www.johnsmith.dev/profile	URL_PERSONAL	Correct
________________________________________
7. Key Findings
The test results identified the following weaknesses of the model.
1.	Phone Number Format Generalization Issue
The model completely missed some U.S. phone number formats or incorrectly classified them as STREET_ADDRESS.
2.	Short Username Detection Issue
Short usernames such as dk123 were not detected because they were difficult to distinguish from ordinary strings.
3.	Email Format Generalization Issue
Even structurally valid email addresses could be missed depending on the specific domain or string pattern.
4.	Difficulty Distinguishing Personal URLs from General URLs
Although the model detects URLs well, there were cases where it could not semantically distinguish personal URLs from general websites.
5.	High Dependence on Context
Entities such as names tended to be detected more reliably when clear PII context such as “My name is” was present nearby.
________________________________________
9. Conclusion
The final DistilBERT model reliably detects PII such as common names, addresses, and personal URLs, but errors may occur with formats that are rare in the training data or with short, ambiguous strings.
In particular, False Negatives were observed with phone number formats, short usernames, and certain email formats, and type confusion also occurred between phone numbers and addresses.
In addition, the case in which a general website was classified as a personal URL confirmed limitations in the semantic distinction of URLs.
These results reveal actual failure patterns of the model that are difficult to identify from a high overall F1 score alone, suggesting the need to expand data diversity and improve generalization to unseen formats in the future.
