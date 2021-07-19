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

class UpdateConfigurationRequest(object):
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
        'config_file_path': 'str',
        'param': 'str',
        'val': 'AnyOfUpdateConfigurationRequestVal'
    }

    attribute_map = {
        'config_file_path': 'configFilePath',
        'param': 'param',
        'val': 'val'
    }

    def __init__(self, config_file_path=None, param=None, val=None):  # noqa: E501
        """UpdateConfigurationRequest - a model defined in Swagger"""  # noqa: E501
        self._config_file_path = None
        self._param = None
        self._val = None
        self.discriminator = None
        if config_file_path is not None:
            self.config_file_path = config_file_path
        self.param = param
        self.val = val

    @property
    def config_file_path(self):
        """Gets the config_file_path of this UpdateConfigurationRequest.  # noqa: E501

        Path to a custom configuration file.  # noqa: E501

        :return: The config_file_path of this UpdateConfigurationRequest.  # noqa: E501
        :rtype: str
        """
        return self._config_file_path

    @config_file_path.setter
    def config_file_path(self, config_file_path):
        """Sets the config_file_path of this UpdateConfigurationRequest.

        Path to a custom configuration file.  # noqa: E501

        :param config_file_path: The config_file_path of this UpdateConfigurationRequest.  # noqa: E501
        :type: str
        """

        self._config_file_path = config_file_path

    @property
    def param(self):
        """Gets the param of this UpdateConfigurationRequest.  # noqa: E501

        Parameter to update name.  # noqa: E501

        :return: The param of this UpdateConfigurationRequest.  # noqa: E501
        :rtype: str
        """
        return self._param

    @param.setter
    def param(self, param):
        """Sets the param of this UpdateConfigurationRequest.

        Parameter to update name.  # noqa: E501

        :param param: The param of this UpdateConfigurationRequest.  # noqa: E501
        :type: str
        """
        if param is None:
            raise ValueError("Invalid value for `param`, must not be `None`")  # noqa: E501

        self._param = param

    @property
    def val(self):
        """Gets the val of this UpdateConfigurationRequest.  # noqa: E501

        Parameter to update value.  # noqa: E501

        :return: The val of this UpdateConfigurationRequest.  # noqa: E501
        :rtype: AnyOfUpdateConfigurationRequestVal
        """
        return self._val

    @val.setter
    def val(self, val):
        """Sets the val of this UpdateConfigurationRequest.

        Parameter to update value.  # noqa: E501

        :param val: The val of this UpdateConfigurationRequest.  # noqa: E501
        :type: AnyOfUpdateConfigurationRequestVal
        """
        if val is None:
            raise ValueError("Invalid value for `val`, must not be `None`")  # noqa: E501

        self._val = val

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
        if issubclass(UpdateConfigurationRequest, dict):
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
        if not isinstance(other, UpdateConfigurationRequest):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other