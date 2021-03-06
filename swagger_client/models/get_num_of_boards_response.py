# coding: utf-8

"""
    GSI Floating-Point 32 API

    **Introduction**<br> GSI Technology’s floating-point similarity search API provides an accessible gateway to running searches on GSI’s Gemini® Associative Processing Unit (APU).<br> It works in conjunction with the GSI system management solution which enables users to work with multiple APU boards simultaneously for improved performance.<br><br> **Dataset and Query Format**<br> Dataset embeddings can be in 32- or 64-bit floating point format, and any number of features, e.g. 256 or 512 (there is no upper limit).<br> Query embeddings must have the same floating-point format and number of features as used in the dataset.<br> GSI performs the search and delivers the top-k most similar results.  # noqa: E501

    OpenAPI spec version: 1.0
    
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""

import pprint
import re  # noqa: F401

import six

class GetNumOfBoardsResponse(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """
    """
    Attributes:
      swagger_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    swagger_types = {
        'total_num_of_boards': 'int',
        'num_of_available_boards': 'int',
        'num_of_allocated_boards': 'int'
    }

    attribute_map = {
        'total_num_of_boards': 'totalNumOfBoards',
        'num_of_available_boards': 'numOfAvailableBoards',
        'num_of_allocated_boards': 'numOfAllocatedBoards'
    }

    def __init__(self, total_num_of_boards=None, num_of_available_boards=None, num_of_allocated_boards=None):  # noqa: E501
        """GetNumOfBoardsResponse - a model defined in Swagger"""  # noqa: E501
        self._total_num_of_boards = None
        self._num_of_available_boards = None
        self._num_of_allocated_boards = None
        self.discriminator = None
        if total_num_of_boards is not None:
            self.total_num_of_boards = total_num_of_boards
        if num_of_available_boards is not None:
            self.num_of_available_boards = num_of_available_boards
        if num_of_allocated_boards is not None:
            self.num_of_allocated_boards = num_of_allocated_boards

    @property
    def total_num_of_boards(self):
        """Gets the total_num_of_boards of this GetNumOfBoardsResponse.  # noqa: E501

        Total number of boards installed on server.  # noqa: E501

        :return: The total_num_of_boards of this GetNumOfBoardsResponse.  # noqa: E501
        :rtype: int
        """
        return self._total_num_of_boards

    @total_num_of_boards.setter
    def total_num_of_boards(self, total_num_of_boards):
        """Sets the total_num_of_boards of this GetNumOfBoardsResponse.

        Total number of boards installed on server.  # noqa: E501

        :param total_num_of_boards: The total_num_of_boards of this GetNumOfBoardsResponse.  # noqa: E501
        :type: int
        """

        self._total_num_of_boards = total_num_of_boards

    @property
    def num_of_available_boards(self):
        """Gets the num_of_available_boards of this GetNumOfBoardsResponse.  # noqa: E501

        Total number of available boards.  # noqa: E501

        :return: The num_of_available_boards of this GetNumOfBoardsResponse.  # noqa: E501
        :rtype: int
        """
        return self._num_of_available_boards

    @num_of_available_boards.setter
    def num_of_available_boards(self, num_of_available_boards):
        """Sets the num_of_available_boards of this GetNumOfBoardsResponse.

        Total number of available boards.  # noqa: E501

        :param num_of_available_boards: The num_of_available_boards of this GetNumOfBoardsResponse.  # noqa: E501
        :type: int
        """

        self._num_of_available_boards = num_of_available_boards

    @property
    def num_of_allocated_boards(self):
        """Gets the num_of_allocated_boards of this GetNumOfBoardsResponse.  # noqa: E501

        Total number of allocated boards  # noqa: E501

        :return: The num_of_allocated_boards of this GetNumOfBoardsResponse.  # noqa: E501
        :rtype: int
        """
        return self._num_of_allocated_boards

    @num_of_allocated_boards.setter
    def num_of_allocated_boards(self, num_of_allocated_boards):
        """Sets the num_of_allocated_boards of this GetNumOfBoardsResponse.

        Total number of allocated boards  # noqa: E501

        :param num_of_allocated_boards: The num_of_allocated_boards of this GetNumOfBoardsResponse.  # noqa: E501
        :type: int
        """

        self._num_of_allocated_boards = num_of_allocated_boards

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value
        if issubclass(GetNumOfBoardsResponse, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, GetNumOfBoardsResponse):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
