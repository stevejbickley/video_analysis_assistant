##### ------ IMPORTS + SETUP ------- ####

import os
import subprocess
import glob
import ffmpeg
import pandas as pd
import numpy as np
import json
import os
import re
import time
from openai import OpenAI
import openpyxl
import requests
import cv2
import base64
from pydantic import BaseModel
import fitz # pip install PyMuPDF

# See reference github repo:  https://github.com/pixegami/openai-assistants-api-demo
# See also OpenAI reference documentation:  ttps://platform.openai.com/docs/assistants/how-it-works

# Enter your Assistant ID here.
ASSISTANT_ID = "asst_NJ580vd7N4ETnei4zI4LEqlZ" # Old (superseded on 10 April - accidental delete): "asst_wWt15CA9kKqTI79SLQDPlGWm"

# Make sure your API key is set as an environment variable.
client = OpenAI()

############ SETUP
CURR_PATH = os.getcwd()
OUTPUT_PATH = CURR_PATH + '/data/output/'
INPUT_PATH = CURR_PATH + '/data/input/'

##### ------ DEFINE FUNCTIONS ------- ####

# For use with assistant (if created)
def create_run(assistant_id, thread_id, message_content):
    # Add message to the thread
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=message_content,
    )
    # Create the run
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )
    return run


# For use with assistant (if created)
def wait_on_run(run, thread_id):
    while run['status'] in ["queued", "in_progress"]:
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run['id'],
        )
        time.sleep(0.5)
    return run


# Function to upload file to openAI
def upload_file(file_path, purpose, api_key):
    """
    Uploads a file to OpenAI and returns the file ID.
    Args:
    - file_path (str): Path to the file to upload.
    - purpose (str): The intended purpose of the uploaded file (e.g., "assistants").
    - api_key (str): Your OpenAI API key.
    Returns:
    - str: The ID of the uploaded file.
    """
    with open(file_path, 'rb') as file:
        response = requests.post(
            "https://api.openai.com/v1/files",
            headers={
                "Authorization": f"Bearer {api_key}"
            },
            files={
                "file": file
            },
            data={
                "purpose": purpose
            }
        )
    response.raise_for_status()
    file_id = response.json()["id"]
    return file_id


# Currently unused function
def convert_pdf_to_base64(file_path):
    """
    Converts a file to base64 encoding.
    Args:
        file_path (str): The path to the file.
    Returns:
        str: The base64 encoded string of the file.
    """
    with open(file_path, "rb") as pdf_file:
        encoded_string = base64.b64encode(pdf_file.read())
    return encoded_string


# Define JSON schema to include fields corresponding to each typology and a description based on the input
json_schema = {
    "type": "object",
    "properties": {
        "False_Connection": {
            "type": "string",
            "description": "Headlines, images, or captions that fail to complement the information."
        },
        "Misleading_Content": {
            "type": "string",
            "description": "Manipulative use of information to create a narrative or target."
        },
        "False_Context": {
            "type": "string",
            "description": "Authentic content shared with false contextual information."
        },
        "Fabrication": {
            "type": "string",
            "description": "Content that is completely fabricated with intent to deceive."
        },
        "Satire_or_Parody": {
            "type": "string",
            "description": "Satirical content misrepresented as fact."
        },
        "Manipulation": {
            "type": "string",
            "description": "Content spun to deceive."
        },
        "Propaganda": {
            "type": "string",
            "description": "Content designed to influence public opinion for political purposes."
        },
        "Biased": {
            "type": "string",
            "description": "One-sided interpretations of data."
        },
        "Inaccurate_Information": {
            "type": "string",
            "description": "Factual errors in the content."
        },
        "Unsubstantiated_Claims": {
            "type": "string",
            "description": "Claims made without journal evidence."
        },
        "Exaggeration": {
            "type": "string",
            "description": "Overstatements or embellishments of the journal's findings."
        }
    },
    "required": [
        "False_Connection",
        "Misleading_Content",
        "False_Context",
        "Fabrication",
        "Satire_or_Parody",
        "Manipulation",
        "Propaganda",
        "Biased",
        "Inaccurate_Information",
        "Unsubstantiated_Claims",
        "Exaggeration"
    ],
    "additionalProperties": False
}


# Function to analyse with strings
def analyze_misinformation_with_texts(news_article, scientific_article, api_key, model="gpt-4o-mini"):
    """
    Analyzes a news article for misinformation based on a scientific article and categorizes it
    using a structured typology of misinformation.
    Args:
    - news_article (str): The text of the news article to analyze.
    - scientific_article (str): The text of the original scientific article.
    - api_key (str): Your OpenAI API key.
    - model (str): The OpenAI model to use.
    Returns:
    - dict: A structured response categorizing the misinformation in the news article.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    prompt = (
        f"Given the original scientific article: {scientific_article}\n\n"
        f"and the following news article: {news_article}\n\n"
        f"Identify and categorize any misinformation present in the news article "
        f"according to the following typology:\n\n"
        f"False Connection, Misleading Content, False Context, Fabrication, "
        f"Satire or Parody, Manipulation, Propaganda, Biased, Inaccurate Information, "
        f"Unsubstantiated Claims, Exaggeration."
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "MisinformationTypologyAnalyzer",
                "strict": True,
                "schema": json_schema
            }
        },
        "max_tokens": 1000
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()


# Function to analyse with pdfs directly
# See e.g.: https://stackoverflow.com/questions/78084538/openai-assistants-api-how-do-i-upload-a-file-and-use-it-as-a-knowledge-base
def analyze_misinformation_with_files(original_pdf_path, media_pdf_path, api_key, model="gpt-4o-mini"):
    """
    Analyzes a news article for misinformation based on a scientific article using uploaded PDF files.
    Args:
    - original_pdf_path (str): Path to the original scientific article PDF.
    - media_pdf_path (str): Path to the news article PDF.
    - api_key (str): Your OpenAI API key.
    - model (str): The OpenAI model to use.
    Returns:
    - dict: A structured response categorizing the misinformation in the news article.
    """
    # Upload the original and media PDFs
    original_file_id = upload_file(original_pdf_path, "assistants", api_key)
    media_file_id = upload_file(media_pdf_path, "assistants", api_key)
    # Construct the prompt referring to the uploaded files
    prompt = (
        f"Given the original scientific article uploaded as {original_file_id} "
        f"and the news article uploaded as {media_file_id}, identify and categorize any misinformation present "
        f"in the news article according to the following typology:\n\n"
        f"False Connection, Misleading Content, False Context, Fabrication, "
        f"Satire or Parody, Manipulation, Propaganda, Biased, Inaccurate Information, "
        f"Unsubstantiated Claims, Exaggeration."
    )
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "file_ids"=[original_file_id,media_file_id]}}],
        "response_format": {
                "name": "MisinformationTypologyAnalyzer",
                "strict": True,
                "schema": json_schema
            },
        "max_tokens": 1000
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    return response.json()


def parse_response_content(content):
    """
    Parses the JSON-like string output from the response and structures it into a list of dictionaries.
    Args:
    - content (str): The JSON-like string from the API response.
    Returns:
    - pd.DataFrame: A dataframe with the structured data or an empty dataframe if parsing fails.
    """
    try:
        # Parse the JSON string into a dictionary
        data = json.loads(content)
        # Attempt to convert the dictionary to a DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # Attempt to create a DataFrame from the dictionary
            try:
                df = pd.DataFrame.from_dict(data, orient='index').reset_index()
            except ValueError:
                # Handle case where the dictionary has scalar values
                df = pd.DataFrame([data])
        else:
            raise ValueError("The parsed data is neither a list nor a dictionary")
    except ValueError as ve:
        print(f"ValueError encountered: {ve}")
        # Return an empty DataFrame to maintain consistency
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred: {e}")
        # Return an empty DataFrame to maintain consistency
        return pd.DataFrame()
    return df


# Example usage with_texts
news_article = "Example news article text..."
#news_article = "Sen. Maggie Hassan, D-N.H., asked whether more safeguards should be put in place on U.S. research grants to gain more insight into how they are being used. Sen. Maggie Hassan, D-N.H., asked whether more safeguards should be put in place on U.S. research grants to gain more insight into how they are being used. SAMUEL CORUM / Getty Images Sponsor Message Get the latest on need-to-know topics for federal employees delivered to your inbox. email Enter your email View Privacy Policy Stay Connected Sponsor Message Featured eBooks Delivering for America: Stamped for Approval or Return to Sender? Cyber Workforce Missile Defense Insights & Reports Fortifying the Force: Robust Resilience in Military Installations Presented By Gordian News Scientists argue over the origins of COVID-19 before Senate panel Microbiology and biodefense experts continued to wrangle over whether COVID-19 emerged from a laboratory leak or was passed to humans through exposure from another animal. June 20, 2024 Homeland Security By Jennifer Shutt Missouri Independent Scientists debated the origins of COVID-19 on Tuesday, trading barbs over whether the bulk of evidence available points to a natural spillover event from a wild animal or a virus designed in a lab and then let loose through an inadvertent leak. The hearing in front of the Senate Homeland Security and Governmental Affairs Committee was part of ongoing efforts in Congress to apply the lessons learned during the pandemic to prevent or blunt the next outbreak. Gregory Koblentz, associate professor and director of the Biodefense Graduate Program at George Mason University in Virginia, said during the two-hour hearing that debate continues in the scientific community about the origins. “The possibility that SARS-CoV-2 was deliberately developed as a biological weapon has been unanimously rejected by all U.S. intelligence agencies,” Koblentz testified. “While the intelligence community is divided on the origin of the pandemic, most of the agencies have determined that the virus was not genetically engineered.” Residents in Wuhan, China, were first diagnosed with “an atypical pneumonia-like illness” in December 2019, according to a COVID-19 timeline from the Centers for Disease Control and Prevention. Initial cases all appeared linked to the Huanan Seafood Wholesale Market at the time, though there has since been much speculation about the types of research taking place at the Wuhan Institute of Virology. Koblentz said he believes the available evidence points to a spillover event from an animal, though he added a “research-related accident can’t be ruled out at this time.” The lack of transparency and data from the Chinese government has significantly hindered scientists’ efforts to unify around the origin of COVID-19, he said. Scientists battle over lab vs. spillover Richard Ebright, board of governors professor of chemistry and chemical biology and laboratory director at the Waksman Institute of Microbiology at Rutgers University in New Jersey, testified he believes a “large preponderance of evidence indicates SARS-CoV-2, the virus that causes COVID-19, entered humans through a research incident.” Ebright also leveled criticism at fellow panelist Robert Garry, who, along with a handful of co-authors, published an opinion article in the journal Nature Medicine in March 2020, titled “The proximal origin of SARS-CoV-2.” In the commentary, Garry and the other scientists wrote, “we do not believe that any type of laboratory-based scenario is plausible.” Ebright said during Tuesday’s hearing that the opinion article represented “scientific misconduct up to and including fraud,” a characterization that Garry rejected during the hearing. “The authors were stating their opinion, but that opinion was not well-founded,” Ebright said. “In March of 2020, there was no basis to state that as a conclusion, as opposed to simply being a hypothesis.” Garry, professor and associate dean of the School of Medicine at Tulane University in Louisiana, argued on behalf of the spillover event during the hearing, testifying that the virus likely didn’t move directly from a bat to humans, but went to an unidentified intermediary animal. “The bat coronaviruses are viruses that are spread by the gastrointestinal route,” Garry said. “For a virus like this to become a respiratory virus — it’s just going to require too many mutations, too many changes for a bat virus to spill directly over to a human being. That could only really happen in nature with replication through an intermediate animal.” Garry also defended gain-of-function research during the hearing, arguing that it has had some beneficial impact, though he noted that it does need “appropriate safeguards and restrictions.” Lawmakers and pundits have used several, often evolving, definitions for gain-of-function research in the wake of the COVID-19 pandemic. The American Society for Microbiology defines it as techniques “used in research to alter the function of an organism in such a way that it is able to do more than it used to do.” When research is “responsibly performed” on highly transmissible and pathogenic viruses, it can lead to advances in public health and national security, Garry testified. “Without gain-of-function research, we’d have no Tamiflu. Without gain-of-function research, we wouldn’t have a vaccine to prevent cancer caused by infection by the human papilloma virus,” Garry said. “And without gain-of-function research, we won’t be able to identify how novel viruses infect us. And if we don’t know how they infect us, we cannot develop appropriate treatments and cures for the next potential pandemic creating virus.” Oversight of funding, research Sen. Maggie Hassan, D-N.H., raised several questions about whether there’s enough oversight of how the United States spends research dollars as well as what mechanisms are in place to monitor how private entities conduct certain types of research. “While their research has the potential to cure diseases and boost our economy, unless they accept federal funding, there is very little federal oversight to ensure that private labs are engaged in safe and ethical research,” she said. Koblentz from George Mason University said there is much less oversight of biosafety and biosecurity for private research facilities that don’t receive federal funding. “In order to expand the scope of oversight to all privately funded research, [it] would require legislative action,” Koblentz said. Congress, he said, should establish a national bio-risk management agency that would have authority over biosafety and biosecurity “regardless of the source of funding.” “At the end of the day, it shouldn’t matter where the funding comes from in terms of making sure this research is being done safely, securely and responsibly,” Koblentz said. Sen. Rand Paul, R-Ky., ranking member on the committee, said the panel will hold an upcoming hearing specifically on gain-of-function research, including what steps Congress should take to ensure it doesn’t put the public at risk. The next pandemic Committee Chairman Gary Peters, D-Mich., said during the hearing that lawmakers “must learn from the challenges faced during this pandemic to ensure we can better protect Americans from future potential biological incidents.” “Our government needs the flexibility to determine the origins of naturally occurring outbreaks, as well as potential outbreaks that could arise from mistakes or malicious intent,” Peters said. Sen. Mitt Romney, R-Utah, after listening to some of the debate, expressed exasperation that so much attention is going toward what caused the last pandemic and not on how to prepare for the next one. “Given the fact that it could have been either, we know what action we ought to take to protect from either,” Romney said. “And so why there’s so much passion around that makes me think it’s more political than scientific, but maybe I’m wrong.” The United States, he said, shouldn’t be funding gain-of-function research and should “insist” that anyone who receives federal funding follow the standards of the International Organization for Standardization. Missouri Independent is part of States Newsroom, a nonprofit news network supported by grants and a coalition of donors as a 501c(3) public charity. Missouri Independent maintains editorial independence. Contact Editor Jason Hancock for questions: info@missouriindependent.com. Follow Missouri Independent on Facebook and X. Share This:"
scientific_article = "Example scientific article text..."
#scientific_article = "Correspondence Published: 17 March 2020 The proximal origin of SARS-CoV-2 Kristian G. Andersen, Andrew Rambaut, W. Ian Lipkin, Edward C. Holmes & Robert F. Garry Nature Medicine volume 26, pages450–452 (2020)Cite this article 5.95m Accesses 34308 Altmetric Metricsdetails To the Editor — Since the first reports of novel pneumonia (COVID-19) in Wuhan, Hubei province, China1,2, there has been considerable discussion on the origin of the causative virus, SARS-CoV-23 (also referred to as HCoV-19)4. Infections with SARS-CoV-2 are now widespread, and as of 11 March 2020, 121,564 cases have been confirmed in more than 110 countries, with 4,373 deaths5. SARS-CoV-2 is the seventh coronavirus known to infect humans; SARS-CoV, MERS-CoV and SARS-CoV-2 can cause severe disease, whereas HKU1, NL63, OC43 and 229E are associated with mild symptoms6. Here we review what can be deduced about the origin of SARS-CoV-2 from comparative analysis of genomic data. We offer a perspective on the notable features of the SARS-CoV-2 genome and discuss scenarios by which they could have arisen. Our analyses clearly show that SARS-CoV-2 is not a laboratory construct or a purposefully manipulated virus. Notable features of the SARS-CoV-2 genome Our comparison of alpha- and betacoronaviruses identifies two notable genomic features of SARS-CoV-2: (i) on the basis of structural studies7,8,9 and biochemical experiments1,9,10, SARS-CoV-2 appears to be optimized for binding to the human receptor ACE2; and (ii) the spike protein of SARS-CoV-2 has a functional polybasic (furin) cleavage site at the S1–S2 boundary through the insertion of 12 nucleotides8, which additionally led to the predicted acquisition of three O-linked glycans around the site. 1. Mutations in the receptor-binding domain of SARS-CoV-2 The receptor-binding domain (RBD) in the spike protein is the most variable part of the coronavirus genome1,2. Six RBD amino acids have been shown to be critical for binding to ACE2 receptors and for determining the host range of SARS-CoV-like viruses7. With coordinates based on SARS-CoV, they are Y442, L472, N479, D480, T487 and Y4911, which correspond to L455, F486, Q493, S494, N501 and Y505 in SARS-CoV-27. Five of these six residues differ between SARS-CoV-2 and SARS-CoV (Fig. 1a). On the basis of structural studies7,8,9 and biochemical experiments1,9,10, SARS-CoV-2 seems to have an RBD that binds with high affinity to ACE2 from humans, ferrets, cats and other species with high receptor homology7. Fig. 1: Features of the spike protein in human SARS-CoV-2 and related coronaviruses. figure 1 a, Mutations in contact residues of the SARS-CoV-2 spike protein. The spike protein of SARS-CoV-2 (red bar at top) was aligned against the most closely related SARS-CoV-like coronaviruses and SARS-CoV itself. Key residues in the spike protein that make contact to the ACE2 receptor are marked with blue boxes in both SARS-CoV-2 and related viruses, including SARS-CoV (Urbani strain). b, Acquisition of polybasic cleavage site and O-linked glycans. Both the polybasic cleavage site and the three adjacent predicted O-linked glycans are unique to SARS-CoV-2 and were not previously seen in lineage B betacoronaviruses. Sequences shown are from NCBI GenBank, accession codes MN908947, MN996532, AY278741, KY417146 and MK211376. The pangolin coronavirus sequences are a consensus generated from SRR10168377 and SRR10168378 (NCBI BioProject PRJNA573298)29,30. Full size image While the analyses above suggest that SARS-CoV-2 may bind human ACE2 with high affinity, computational analyses predict that the interaction is not ideal7 and that the RBD sequence is different from those shown in SARS-CoV to be optimal for receptor binding7,11. Thus, the high-affinity binding of the SARS-CoV-2 spike protein to human ACE2 is most likely the result of natural selection on a human or human-like ACE2 that permits another optimal binding solution to arise. This is strong evidence that SARS-CoV-2 is not the product of purposeful manipulation. 2. Polybasic furin cleavage site and O-linked glycans The second notable feature of SARS-CoV-2 is a polybasic cleavage site (RRAR) at the junction of S1 and S2, the two subunits of the spike8 (Fig. 1b). This allows effective cleavage by furin and other proteases and has a role in determining viral infectivity and host range12. In addition, a leading proline is also inserted at this site in SARS-CoV-2; thus, the inserted sequence is PRRA (Fig. 1b). The turn created by the proline is predicted to result in the addition of O-linked glycans to S673, T678 and S686, which flank the cleavage site and are unique to SARS-CoV-2 (Fig. 1b). Polybasic cleavage sites have not been observed in related ‘lineage B’ betacoronaviruses, although other human betacoronaviruses, including HKU1 (lineage A), have those sites and predicted O-linked glycans13. Given the level of genetic variation in the spike, it is likely that SARS-CoV-2-like viruses with partial or full polybasic cleavage sites will be discovered in other species. The functional consequence of the polybasic cleavage site in SARS-CoV-2 is unknown, and it will be important to determine its impact on transmissibility and pathogenesis in animal models. Experiments with SARS-CoV have shown that insertion of a furin cleavage site at the S1–S2 junction enhances cell–cell fusion without affecting viral entry14. In addition, efficient cleavage of the MERS-CoV spike enables MERS-like coronaviruses from bats to infect human cells15. In avian influenza viruses, rapid replication and transmission in highly dense chicken populations selects for the acquisition of polybasic cleavage sites in the hemagglutinin (HA) protein16, which serves a function similar to that of the coronavirus spike protein. Acquisition of polybasic cleavage sites in HA, by insertion or recombination, converts low-pathogenicity avian influenza viruses into highly pathogenic forms16. The acquisition of polybasic cleavage sites by HA has also been observed after repeated passage in cell culture or through animals17. The function of the predicted O-linked glycans is unclear, but they could create a ‘mucin-like domain’ that shields epitopes or key residues on the SARS-CoV-2 spike protein18. Several viruses utilize mucin-like domains as glycan shields involved immunoevasion18. Although prediction of O-linked glycosylation is robust, experimental studies are needed to determine if these sites are used in SARS-CoV-2. Theories of SARS-CoV-2 origins It is improbable that SARS-CoV-2 emerged through laboratory manipulation of a related SARS-CoV-like coronavirus. As noted above, the RBD of SARS-CoV-2 is optimized for binding to human ACE2 with an efficient solution different from those previously predicted7,11. Furthermore, if genetic manipulation had been performed, one of the several reverse-genetic systems available for betacoronaviruses would probably have been used19. However, the genetic data irrefutably show that SARS-CoV-2 is not derived from any previously used virus backbone20. Instead, we propose two scenarios that can plausibly explain the origin of SARS-CoV-2: (i) natural selection in an animal host before zoonotic transfer; and (ii) natural selection in humans following zoonotic transfer. We also discuss whether selection during passage could have given rise to SARS-CoV-2. 1. Natural selection in an animal host before zoonotic transfer As many early cases of COVID-19 were linked to the Huanan market in Wuhan1,2, it is possible that an animal source was present at this location. Given the similarity of SARS-CoV-2 to bat SARS-CoV-like coronaviruses2, it is likely that bats serve as reservoir hosts for its progenitor. Although RaTG13, sampled from a Rhinolophus affinis bat1, is ~96% identical overall to SARS-CoV-2, its spike diverges in the RBD, which suggests that it may not bind efficiently to human ACE27 (Fig. 1a). Malayan pangolins (Manis javanica) illegally imported into Guangdong province contain coronaviruses similar to SARS-CoV-221. Although the RaTG13 bat virus remains the closest to SARS-CoV-2 across the genome1, some pangolin coronaviruses exhibit strong similarity to SARS-CoV-2 in the RBD, including all six key RBD residues21 (Fig. 1). This clearly shows that the SARS-CoV-2 spike protein optimized for binding to human-like ACE2 is the result of natural selection. Neither the bat betacoronaviruses nor the pangolin betacoronaviruses sampled thus far have polybasic cleavage sites. Although no animal coronavirus has been identified that is sufficiently similar to have served as the direct progenitor of SARS-CoV-2, the diversity of coronaviruses in bats and other species is massively undersampled. Mutations, insertions and deletions can occur near the S1–S2 junction of coronaviruses22, which shows that the polybasic cleavage site can arise by a natural evolutionary process. For a precursor virus to acquire both the polybasic cleavage site and mutations in the spike protein suitable for binding to human ACE2, an animal host would probably have to have a high population density (to allow natural selection to proceed efficiently) and an ACE2-encoding gene that is similar to the human ortholog. 2. Natural selection in humans following zoonotic transfer It is possible that a progenitor of SARS-CoV-2 jumped into humans, acquiring the genomic features described above through adaptation during undetected human-to-human transmission. Once acquired, these adaptations would enable the pandemic to take off and produce a sufficiently large cluster of cases to trigger the surveillance system that detected it1,2. All SARS-CoV-2 genomes sequenced so far have the genomic features described above and are thus derived from a common ancestor that had them too. The presence in pangolins of an RBD very similar to that of SARS-CoV-2 means that we can infer this was also probably in the virus that jumped to humans. This leaves the insertion of polybasic cleavage site to occur during human-to-human transmission. Estimates of the timing of the most recent common ancestor of SARS-CoV-2 made with current sequence data point to emergence of the virus in late November 2019 to early December 201923, compatible with the earliest retrospectively confirmed cases24. Hence, this scenario presumes a period of unrecognized transmission in humans between the initial zoonotic event and the acquisition of the polybasic cleavage site. Sufficient opportunity could have arisen if there had been many prior zoonotic events that produced short chains of human-to-human transmission over an extended period. This is essentially the situation for MERS-CoV, for which all human cases are the result of repeated jumps of the virus from dromedary camels, producing single infections or short transmission chains that eventually resolve, with no adaptation to sustained transmission25. Studies of banked human samples could provide information on whether such cryptic spread has occurred. Retrospective serological studies could also be informative, and a few such studies have been conducted showing low-level exposures to SARS-CoV-like coronaviruses in certain areas of China26. Critically, however, these studies could not have distinguished whether exposures were due to prior infections with SARS-CoV, SARS-CoV-2 or other SARS-CoV-like coronaviruses. Further serological studies should be conducted to determine the extent of prior human exposure to SARS-CoV-2. 3. Selection during passage Basic research involving passage of bat SARS-CoV-like coronaviruses in cell culture and/or animal models has been ongoing for many years in biosafety level 2 laboratories across the world27, and there are documented instances of laboratory escapes of SARS-CoV28. We must therefore examine the possibility of an inadvertent laboratory release of SARS-CoV-2. In theory, it is possible that SARS-CoV-2 acquired RBD mutations (Fig. 1a) during adaptation to passage in cell culture, as has been observed in studies of SARS-CoV11. The finding of SARS-CoV-like coronaviruses from pangolins with nearly identical RBDs, however, provides a much stronger and more parsimonious explanation of how SARS-CoV-2 acquired these via recombination or mutation19. The acquisition of both the polybasic cleavage site and predicted O-linked glycans also argues against culture-based scenarios. New polybasic cleavage sites have been observed only after prolonged passage of low-pathogenicity avian influenza virus in vitro or in vivo17. Furthermore, a hypothetical generation of SARS-CoV-2 by cell culture or animal passage would have required prior isolation of a progenitor virus with very high genetic similarity, which has not been described. Subsequent generation of a polybasic cleavage site would have then required repeated passage in cell culture or animals with ACE2 receptors similar to those of humans, but such work has also not previously been described. Finally, the generation of the predicted O-linked glycans is also unlikely to have occurred due to cell-culture passage, as such features suggest the involvement of an immune system18. Conclusions In the midst of the global COVID-19 public-health emergency, it is reasonable to wonder why the origins of the pandemic matter. Detailed understanding of how an animal virus jumped species boundaries to infect humans so productively will help in the prevention of future zoonotic events. For example, if SARS-CoV-2 pre-adapted in another animal species, then there is the risk of future re-emergence events. In contrast, if the adaptive process occurred in humans, then even if repeated zoonotic transfers occur, they are unlikely to take off without the same series of mutations. In addition, identifying the closest viral relatives of SARS-CoV-2 circulating in animals will greatly assist studies of viral function. Indeed, the availability of the RaTG13 bat sequence helped reveal key RBD mutations and the polybasic cleavage site. The genomic features described here may explain in part the infectiousness and transmissibility of SARS-CoV-2 in humans. Although the evidence shows that SARS-CoV-2 is not a purposefully manipulated virus, it is currently impossible to prove or disprove the other theories of its origin described here. However, since we observed all notable SARS-CoV-2 features, including the optimized RBD and polybasic cleavage site, in related coronaviruses in nature, we do not believe that any type of laboratory-based scenario is plausible. More scientific data could swing the balance of evidence to favor one hypothesis over another. Obtaining related viral sequences from animal sources would be the most definitive way of revealing viral origins. For example, a future observation of an intermediate or fully formed polybasic cleavage site in a SARS-CoV-2-like virus from animals would lend even further support to the natural-selection hypotheses. It would also be helpful to obtain more genetic and functional data about SARS-CoV-2, including animal studies. The identification of a potential intermediate host of SARS-CoV-2, as well as sequencing of the virus from very early cases, would similarly be highly informative. Irrespective of the exact mechanisms by which SARS-CoV-2 originated via natural selection, the ongoing surveillance of pneumonia in humans and other animals is clearly of utmost importance. References Zhou, P. et al. Nature https://doi.org/10.1038/s41586-020-2012-7 (2020). Article PubMed PubMed Central Google Scholar Wu, F. et al. Nature https://doi.org/10.1038/s41586-020-2008-3 (2020). Article PubMed PubMed Central Google Scholar Gorbalenya, A. E. et al. bioRxiv https://doi.org/10.1101/2020.02.07.937862 (2020). Article Google Scholar Jiang, S. et al. Lancet https://doi.org/10.1016/S0140-6736(20)30419-0 (2020). Article PubMed PubMed Central Google Scholar Dong, E., Du, H. & Gardner, L. Lancet Infect. Dis. https://doi.org/10.1016/S1473-3099(20)30120-1 (2020). Article PubMed Google Scholar Corman, V. M., Muth, D., Niemeyer, D. & Drosten, C. Adv. Virus Res. 100, 163–188 (2018). Article Google Scholar Wan, Y., Shang, J., Graham, R., Baric, R. S. & Li, F. J. Virol. https://doi.org/10.1128/JVI.00127-20 (2020). Article PubMed Google Scholar Walls, A. C. et al. bioRxiv https://doi.org/10.1101/2020.02.19.956581 (2020). Article Google Scholar Wrapp, D. et al. Science https://doi.org/10.1126/science.abb2507 (2020). Article PubMed Google Scholar Letko, M., Marzi, A. & Munster, V. Nat. Microbiol. https://doi.org/10.1038/s41564-020-0688-y (2020). Article PubMed PubMed Central Google Scholar Sheahan, T. et al. J. Virol. 82, 2274–2285 (2008). Article CAS Google Scholar Nao, N. et al. MBio 8, e02298-16 (2017). Article Google Scholar Chan, C.-M. et al. Exp. Biol. Med. 233, 1527–1536 (2008). Article CAS Google Scholar Follis, K. E., York, J. & Nunberg, J. H. Virology 350, 358–369 (2006). Article CAS Google Scholar Menachery, V. D. et al. J. Virol. https://doi.org/10.1128/JVI.01774-19 (2019). Article PubMed Google Scholar Alexander, D. J. & Brown, I. H. Rev. Sci. Tech. 28, 19–38 (2009). Article CAS Google Scholar Ito, T. et al. J. Virol. 75, 4439–4443 (2001). Article CAS Google Scholar Bagdonaite, I. & Wandall, H. H. Glycobiology 28, 443–467 (2018). Article CAS Google Scholar Cui, J., Li, F. & Shi, Z.-L. Nat. Rev. Microbiol. 17, 181–192 (2019). Article CAS Google Scholar Almazán, F. et al. Virus Res. 189, 262–270 (2014). Article Google Scholar Zhang, T., Wu, Q. & Zhang, Z. bioRxiv https://doi.org/10.1101/2020.02.19.950253 (2020). Article Google Scholar Yamada, Y. & Liu, D. X. J. Virol. 83, 8744–8758 (2009). Article CAS Google Scholar Rambaut, A. Virological.org http://virological.org/t/356 (2020). Huang, C. et al. Lancet https://doi.org/10.1016/S0140-6736(20)30183-5 (2020). Article PubMed PubMed Central Google Scholar Dudas, G., Carvalho, L. M., Rambaut, A. & Bedford, T. eLife 7, e31257 (2018). Article Google Scholar Wang, N. et al. Virol. Sin. 33, 104–107 (2018). Article Google Scholar Ge, X.-Y. et al. Nature 503, 535–538 (2013). Article CAS Google Scholar Lim, P. L. et al. N. Engl. J. Med. 350, 1740–1745 (2004). Article CAS Google Scholar Wong, M. C., Javornik Cregeen, S. J., Ajami, N. J. & Petrosino, J. F. bioRxiv https://doi.org/10.1101/2020.02.07.939207 (2020). Article Google Scholar Liu, P., Chen, W. & Chen, J.-P. Viruses 11, 979 (2019). Article CAS Google Scholar Download references Acknowledgements We thank all those who have contributed sequences to the GISAID database (https://www.gisaid.org/) and analyses to Virological.org (http://virological.org/). We thank M. Farzan for discussions, and the Wellcome Trust for support. K.G.A. is a Pew Biomedical Scholar and is supported by NIH grant U19AI135995. A.R. is supported by the Wellcome Trust (Collaborators Award 206298/Z/17/Z―ARTIC network) and the European Research Council (grant agreement no. 725422―ReservoirDOCS). E.C.H. is supported by an ARC Australian Laureate Fellowship (FL170100022). R.F.G. is supported by NIH grants U19AI135995, U54 HG007480 and U19AI142790. Author information Authors and Affiliations Department of Immunology and Microbiology, The Scripps Research Institute, La Jolla, CA, USA Kristian G. Andersen Scripps Research Translational Institute, La Jolla, CA, USA Kristian G. Andersen Institute of Evolutionary Biology, University of Edinburgh, Edinburgh, UK Andrew Rambaut Center for Infection and Immunity, Mailman School of Public Health of Columbia University, New York, NY, USA W. Ian Lipkin Marie Bashir Institute for Infectious Diseases and Biosecurity, School of Life and Environmental Sciences and School of Medical Sciences, The University of Sydney, Sydney, Australia Edward C. Holmes Tulane University, School of Medicine, Department of Microbiology and Immunology, New Orleans, LA, USA Robert F. Garry Zalgen Labs, Germantown, MD, USA Robert F. Garry Corresponding author Correspondence to Kristian G. Andersen. Ethics declarations Competing interests R.F.G. is co-founder of Zalgen Labs, a biotechnology company that develops countermeasures to emerging viruses. Rights and permissions Reprints and permissions About this article Check for updates. Verify currency and authenticity via CrossMark Cite this article Andersen, K.G., Rambaut, A., Lipkin, W.I. et al. The proximal origin of SARS-CoV-2. Nat Med 26, 450–452 (2020). https://doi.org/10.1038/s41591-020-0820-9 Download citation Published 17 March 2020 Issue Date April 2020 DOI https://doi.org/10.1038/s41591-020-0820-9"

# Use the analyze_misinformation function to get the result
result = analyze_misinformation_with_texts(news_article, scientific_article, os.environ['OPENAI_API_KEY'], model="gpt-4o-2024-08-06")
result_df = parse_response_content(result['choices'][0]['message']['content'])

# Save to CSV
result_df.to_csv(OUTPUT_PATH + 'result_df.csv', index=False)

# Print the categorized misinformation
print(result)

# Example usage with_files
# https://www.altmetric.com/explorer/outputs?scope=all&show_details=77676422
# https://www.nature.com/articles/s41591-020-0820-9
# https://www.govexec.com/federal-news/2024/06/scientists-argue-over-origins-covid-19-senate-panel/397500/
original_pdf_path = "s41591-020-0820-9_original.pdf"
media_pdf_path = "s41591-020-0820-9_media.pdf"

result = analyze_misinformation_with_files(original_pdf_path, media_pdf_path, os.environ['OPENAI_API_KEY'], model="gpt-4o-2024-08-06")

# Print the categorized misinformation
print(result)