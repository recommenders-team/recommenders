# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
#   MicrosoftAcademicGraph class to read MAG streams for Pandas
#
#   Note:
#     MAG streams do not have header
#

import numpy as np
import pandas as pd


class MicrosoftAcademicGraph:

    # constructor
    def __init__(self, root):
        self.root = root

    # return stream path
    def get_full_path(self, stream_name):
        return self.root + stream_name + ".txt"

    # return stream header
    def get_header(self, stream_name):
        return self.streams[stream_name]

    # return stream types and columns with date
    def get_type(self, stream_name):
        date_columns = []
        schema = {}
        for field in self.streams[stream_name]:
            fieldname, fieldtype = field.split(":")
            nullable = fieldtype.endswith("?")
            if nullable:
                fieldtype = fieldtype[:-1]
            if fieldtype == "DateTime":
                date_columns.append(fieldname)
            schema[fieldname] = self.datatypedict[fieldtype]
        return schema, date_columns

    # return stream columns names
    def get_name(self, stream_name):
        names = []
        for field in self.streams[stream_name]:
            fieldname, fieldtype = field.split(":")
            names.append(fieldname)
        return names

    # return stream Pandas dataFrame
    def get_data_frame(self, stream_name):
        column_name = self.get_name(stream_name)
        column_type, date_columns = self.get_type(stream_name)
        return pd.read_csv(
            filepath_or_buffer=self.get_full_path(stream_name),
            parse_dates=date_columns,
            low_memory=False,
            names=column_name,
            dtype=column_type,
            date_parser=self.date_parse_func,
            sep="\t",
        )

    # date parse function
    date_parse_func = lambda self, c: pd.to_datetime(
        c, format="%m/%d/%Y %H:%M:%S %p", errors="coerce"
    )  # 6/24/2016 12:00:00 AM

    # convert input datatype to Pandas datatype
    datatypedict = {
        "int": pd.Int32Dtype(),
        "uint": pd.UInt32Dtype(),
        "long": pd.Int64Dtype(),
        "ulong": pd.UInt64Dtype(),
        "float": np.float32,
        "string": np.string_,
        "DateTime": np.string_,
    }

    # define stream dictionary
    streams = {
        "Affiliations": [
            "AffiliationId:long",
            "Rank:uint",
            "NormalizedName:string",
            "DisplayName:string",
            "GridId:string",
            "OfficialPage:string",
            "WikiPage:string",
            "PaperCount:long",
            "PaperFamilyCount:long",
            "CitationCount:long",
            "Latitude:float?",
            "Longitude:float?",
            "CreatedDate:DateTime",
        ],
        "Authors": [
            "AuthorId:long",
            "Rank:uint",
            "NormalizedName:string",
            "DisplayName:string",
            "LastKnownAffiliationId:long?",
            "PaperCount:long",
            "PaperFamilyCount:long",
            "CitationCount:long",
            "CreatedDate:DateTime",
        ],
        "ConferenceInstances": [
            "ConferenceInstanceId:long",
            "NormalizedName:string",
            "DisplayName:string",
            "ConferenceSeriesId:long",
            "Location:string",
            "OfficialUrl:string",
            "StartDate:DateTime?",
            "EndDate:DateTime?",
            "AbstractRegistrationDate:DateTime?",
            "SubmissionDeadlineDate:DateTime?",
            "NotificationDueDate:DateTime?",
            "FinalVersionDueDate:DateTime?",
            "PaperCount:long",
            "PaperFamilyCount:long",
            "CitationCount:long",
            "Latitude:float?",
            "Longitude:float?",
            "CreatedDate:DateTime",
        ],
        "ConferenceSeries": [
            "ConferenceSeriesId:long",
            "Rank:uint",
            "NormalizedName:string",
            "DisplayName:string",
            "PaperCount:long",
            "PaperFamilyCount:long",
            "CitationCount:long",
            "CreatedDate:DateTime",
        ],
        "EntityRelatedEntities": [
            "EntityId:long",
            "EntityType:string",
            "RelatedEntityId:long",
            "RelatedEntityType:string",
            "RelatedType:int",
            "Score:float",
        ],
        "FieldOfStudyChildren": ["FieldOfStudyId:long", "ChildFieldOfStudyId:long"],
        "FieldOfStudyExtendedAttributes": [
            "FieldOfStudyId:long",
            "AttributeType:int",
            "AttributeValue:string",
        ],
        "FieldsOfStudy": [
            "FieldOfStudyId:long",
            "Rank:uint",
            "NormalizedName:string",
            "DisplayName:string",
            "MainType:string",
            "Level:int",
            "PaperCount:long",
            "PaperFamilyCount:long",
            "CitationCount:long",
            "CreatedDate:DateTime",
        ],
        "Journals": [
            "JournalId:long",
            "Rank:uint",
            "NormalizedName:string",
            "DisplayName:string",
            "Issn:string",
            "Publisher:string",
            "Webpage:string",
            "PaperCount:long",
            "PaperFamilyCount:long",
            "CitationCount:long",
            "CreatedDate:DateTime",
        ],
        "PaperAbstractsInvertedIndex": ["PaperId:long", "IndexedAbstract:string"],
        "PaperAuthorAffiliations": [
            "PaperId:long",
            "AuthorId:long",
            "AffiliationId:long?",
            "AuthorSequenceNumber:uint",
            "OriginalAuthor:string",
            "OriginalAffiliation:string",
        ],
        "PaperCitationContexts": [
            "PaperId:long",
            "PaperReferenceId:long",
            "CitationContext:string",
        ],
        "PaperExtendedAttributes": [
            "PaperId:long",
            "AttributeType:int",
            "AttributeValue:string",
        ],
        "PaperFieldsOfStudy": ["PaperId:long", "FieldOfStudyId:long", "Score:float"],
        "PaperRecommendations": [
            "PaperId:long",
            "RecommendedPaperId:long",
            "Score:float",
        ],
        "PaperReferences": ["PaperId:long", "PaperReferenceId:long"],
        "PaperResources": [
            "PaperId:long",
            "ResourceType:int",
            "ResourceUrl:string",
            "SourceUrl:string",
            "RelationshipType:int",
        ],
        "PaperUrls": [
            "PaperId:long",
            "SourceType:int?",
            "SourceUrl:string",
            "LanguageCode:string",
        ],
        "Papers": [
            "PaperId:long",
            "Rank:uint",
            "Doi:string",
            "DocType:string",
            "PaperTitle:string",
            "OriginalTitle:string",
            "BookTitle:string",
            "Year:int?",
            "Date:DateTime?",
            "OnlineDate:DateTime?",
            "Publisher:string",
            "JournalId:long?",
            "ConferenceSeriesId:long?",
            "ConferenceInstanceId:long?",
            "Volume:string",
            "Issue:string",
            "FirstPage:string",
            "LastPage:string",
            "ReferenceCount:long",
            "CitationCount:long",
            "EstimatedCitation:long",
            "OriginalVenue:string",
            "FamilyId:long?",
            "CreatedDate:DateTime",
        ],
        "RelatedFieldOfStudy": [
            "FieldOfStudyId1:long",
            "Type1:string",
            "FieldOfStudyId2:long",
            "Type2:string",
            "Rank:float",
        ],
    }
